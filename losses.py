import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import misc 

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)): 
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): 
            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):

        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean()


class PartialLogLikelihood(nn.Module):
    """DeepSurv: Personalized Treatment Recommender System Using A Cox Proportional Hazards Deep Neural Network"""
    def __init__(self):
        super(PartialLogLikelihood, self).__init__()
    def forward(self, logits, fail_indicator):
        '''
        fail_indicator: 1 if the sample fails, 0 if the sample is censored.
        logits: raw output from model 
        '''

        log_h = logits.view(-1)
        events = fail_indicator.view(-1)

        log_cumsum_h = log_h.exp().cumsum(0).log()
        return - ((log_h.sub(log_cumsum_h))[events==1]).sum().div(events.sum())

# class PartialLogLikelihood(nn.Module):
#     def __init__(self):
#         super(PartialLogLikelihood, self).__init__()
#     def forward(self, logits, fail_indicator):
#         '''
#         fail_indicator: 1 if the sample fails, 0 if the sample is censored.
#         logits: raw output from model 
#         ties: 'noties' or 'efron' or 'breslow'
#         '''
#         logL = 0

#         # pre-calculate cumsum

#         cumsum_y_pred = torch.cumsum(logits, 0)
#         hazard_ratio = torch.exp(logits)
#         cumsum_hazard_ratio = torch.cumsum(hazard_ratio, 0)
#         log_risk = torch.log(cumsum_hazard_ratio)
#         likelihood = logits - log_risk
#         # dimension for E: np.array -> [None, 1]
#         uncensored_likelihood = likelihood * fail_indicator
#         logL = -torch.sum(uncensored_likelihood)
  
#         # negative average log-likelihood
#         observations = torch.sum(fail_indicator, 0)
#         return 1.0*logL / observations  

# class PartialLogLikelihood(nn.Module):
#     def __init__(self):
#         super(PartialLogLikelihood, self).__init__()
#         self.gamma = -np.inf
#     def forward(self, logits, fail_indicator, eps = 1e-7):
#         '''
#         fail_indicator: 1 if the sample fails, 0 if the sample is censored.
#         logits: raw output from model 
#         '''
        
#         fail_indicator = fail_indicator.view(-1)
#         log_h = logits
#         log_h = log_h.view(-1)
#         if log_h.max() > self.gamma:
#             self.gamma = log_h.max().item()
#         log_cumsum_h = log_h.sub(self.gamma).exp().cumsum(0).add(eps).log().add(self.gamma)
#         return - log_h.sub(log_cumsum_h).mul(fail_indicator).sum().div(fail_indicator.sum()) 


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
    def forward(self, y_true, y_pred):
        loss = (abs(y_true - y_pred)).sum() / y_true.shape[0]
        return loss  

class SoftDiceLoss_v1(nn.Module):
    def __init__(self):  
        super(SoftDiceLoss_v1, self).__init__()

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        smooth = 1.0    
        m1  = probs.view (-1)
        m2  = labels.view(-1)
        intersection = (m1 * m2).sum()
        score = (2.0 * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
        
        return 1- score  


from functools import reduce
def acc_pairs(censor, lifetime, t, time_bin):
    noncensor_index = np.nonzero(censor.data.numpy())[0]
    acc_pair = []
    for i in noncensor_index:
        """didn't consider equal time with second subject censored
        """
        all_j = np.array(range(len(lifetime)))[(lifetime > lifetime[i])* (lifetime[i] > time_bin/7*t)]
        acc_pair.append([(i, j) for j in all_j])

    acc_pair = reduce(lambda x, y: x + y, acc_pair)
    return acc_pair

def rank_loss(lifetime, censor, score2):
    total = 0
    time_bin = 30
    n_pair = 0
    for t in range(128):
        pairs = acc_pairs(censor, lifetime, t,  time_bin)
        n_pair += len(pairs)
        for i, j in pairs:
            L2dist = (score2[j, t] - score2[i, t] -1 )**2
            total += L2dist #* yi * (1-yj)
    return total/ n_pair

def weibull_loss(model, t, e, risk='1'):
    
    shape, scale = model.shape, model.scale
    k_ = shape.expand(t.shape[0], -1)
    b_ = scale.expand(t.shape[0], -1)

    ll = 0.
    for g in range(model.k):

        k = k_[:, g]
        b = b_[:, g]

        s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
        f = k + b + ((torch.exp(k)-1)*(b+torch.log(t)))
        f = f + s

        uncens = np.where(e.cpu().data.numpy() == int(risk))[0]
        cens = np.where(e.cpu().data.numpy() != int(risk))[0]
        ll += f[uncens].sum() + s[cens].sum()

    return -ll.mean()

def conditional_exponient_loss(model, x, t, e, pdf_u, pdf_c, hr_loss=False, imbalance_loss=False, elbo=True, risk=1):
    lossf, losss = [], []
    shape, scale, gates = model(x)
    # if torch.isnan(model(x[-1:])[0]):
    #     print(x)
    shapes, scales = shape.exp(), (-scale).exp()
    loss_neg = 0
    for idx in range(shapes.shape[1]):
        
        eta = shapes[:, idx]
        beta = scales[:, idx]
        #print(eta[0], beta[0])
        log_s = -t*eta
        log_f = torch.log(eta) + log_s

        lossf.append(log_f)
        losss.append(log_s)
        
        # negative partial log likelihood
        hr = torch.log(eta)
        loss_neg += PartialLogLikelihood()(hr, e)
    

    losss = torch.stack(losss, dim=1)
    lossf = torch.stack(lossf, dim=1)

    if elbo:
        lossg = nn.Softmax(dim=1)(gates)
        losss = lossg*losss
        lossf = lossg*lossf
        losss = losss.sum(dim=1)
        lossf = lossf.sum(dim=1)
    else:
        lossg = nn.LogSoftmax(dim=1)(gates)
        losss = lossg + losss
        lossf = lossg + lossf
        losss = torch.logsumexp(losss, dim=1)
        lossf = torch.logsumexp(lossf, dim=1)

    if imbalance_loss:
        try:
            idx_time = t.int().cpu().detach().numpy()
            idx_time[idx_time >=10] = 9
            pdf_u_ = torch.tensor(pdf_u).cuda()
            pdf_c_ = torch.tensor(pdf_c).cuda()
            lossf = lossf * ((1- pdf_u_[idx_time]).exp())
            losss = losss * ((1- pdf_c_[idx_time]).exp())
        except:
            pass
    uncens = np.where(e.cpu().data.numpy() == int(risk))[0]
    cens = np.where(e.cpu().data.numpy() != int(risk))[0]
    ll = lossf[uncens].sum() + model.discount*losss[cens].sum()

    #print(-ll/float(len(uncens)+len(cens)), loss_neg)
    if hr_loss and e.sum() > 0:
        return -ll/float(len(uncens)+len(cens)) + loss_neg*model.gamma
    else:
        return -ll/float(len(uncens)+len(cens))


def conditional_weibull_loss(data_type, model, X_CT, X_PET, X_Clinical, t, e, pdf_u, pdf_c, hr_loss=False, imbalance_loss=False, device=None, elbo=True, risk=1):
    lossf, losss = [], []
    if "ct" == data_type:
        print(device)
        shape, scale, gates = model(X_CT.to(device))
    elif "pet" == data_type:
        shape, scale, gates = model(X_PET.to(device))
    elif "clinical" == data_type:
        shape, scale, gates = model(X_Clinical.to(device))
    # if torch.isnan(model(x[-1:])[0]):
    #     print(x)
    shapes, scales = shape.exp(), (-scale).exp()
    loss_neg = 0
    for idx in range(shapes.shape[1]):
        
        eta = shapes[:, idx]
        beta = scales[:, idx]
        #print(eta[0], beta[0])
        log_s = - (torch.pow(t/beta, eta))
        log_f = torch.log(eta) - torch.log(beta) + ((eta-1)*(-torch.log(beta)+torch.log(t)))
        log_f = log_f + log_s

        lossf.append(log_f)
        losss.append(log_s)
        
        # negative partial log likelihood
        hr = torch.log((eta/beta)*((t/beta)**(eta-1)))
        loss_neg += PartialLogLikelihood()(hr, e)
    

    losss = torch.stack(losss, dim=1)
    lossf = torch.stack(lossf, dim=1)

    if elbo:
        lossg = nn.Softmax(dim=1)(gates)
        losss = lossg*losss
        lossf = lossg*lossf
        losss = losss.sum(dim=1)
        lossf = lossf.sum(dim=1)
    else:
        lossg = nn.LogSoftmax(dim=1)(gates)
        losss = lossg + losss
        lossf = lossg + lossf
        losss = torch.logsumexp(losss, dim=1)
        lossf = torch.logsumexp(lossf, dim=1)

    if imbalance_loss:
        try:
            idx_time = t.int().cpu().detach().numpy()
            idx_time[idx_time>=10]= 9
            pdf_u_ = torch.tensor(pdf_u).cuda()
            pdf_c_ = torch.tensor(pdf_c).cuda()
            lossf = lossf * ((1- pdf_u_[idx_time]).exp())
            losss = losss * ((1- pdf_c_[idx_time]).exp())
        except:
            pass
    uncens = np.where(e.cpu().data.numpy() == int(risk))[0]
    cens = np.where(e.cpu().data.numpy() != int(risk))[0]
    ll = lossf[uncens].sum() + model.discount*losss[cens].sum()

    if hr_loss and e.sum() > 0:
        return -ll/float(len(uncens)+len(cens)) + loss_neg*model.gamma
    else:
        return -ll/float(len(uncens)+len(cens))

def conditional_lognormal_loss(model, x, t, e, pdf_u, pdf_c, hr_loss=False, imbalance_loss=False, elbo=True, risk=1):
        
    shape, scale, logits = model.forward(x)

    lossf = []
    losss = []

    k_ = shape
    b_ = scale
    loss_neg = 0
    for g in range(model.k):

        mu = k_[:, g]
        sigma = b_[:, g]

        f = - sigma - 0.5*np.log(2*np.pi)
        f = f - torch.div((torch.log(t) - mu)**2, 2.*torch.exp(2*sigma))
        s = torch.div(torch.log(t) - mu, torch.exp(sigma)*np.sqrt(2))
        s = 0.5 - 0.5*torch.erf(s)
        s = torch.log(s)

        lossf.append(f)
        losss.append(s)

        # negative partial log likelihood
        hr = f - s
        loss_neg += PartialLogLikelihood()(hr, e)

    losss = torch.stack(losss, dim=1)
    lossf = torch.stack(lossf, dim=1)

    if elbo:
        lossg = nn.Softmax(dim=1)(logits)
        losss = lossg*losss
        lossf = lossg*lossf

        losss = losss.sum(dim=1)
        lossf = lossf.sum(dim=1)
    else:
        lossg = nn.LogSoftmax(dim=1)(logits)
        losss = lossg + losss
        lossf = lossg + lossf

        losss = torch.logsumexp(losss, dim=1)
        lossf = torch.logsumexp(lossf, dim=1)

    if imbalance_loss:
        try:
            idx_time = t.int().cpu().detach().numpy()
            idx_time[idx_time>=10]= 9
            pdf_u_ = torch.tensor(pdf_u).cuda()
            pdf_c_ = torch.tensor(pdf_c).cuda()
            lossf = lossf * ((1- pdf_u_[idx_time]).exp())
            losss = losss * ((1- pdf_c_[idx_time]).exp())
        except:
            pass

    uncens = np.where(e.cpu().data.numpy() == int(risk))[0]
    cens = np.where(e.cpu().data.numpy() != int(risk))[0]
    ll = lossf[uncens].sum() + model.discount*losss[cens].sum()

    if hr_loss and e.sum() > 0:
        return -ll/float(len(uncens)+len(cens)) + loss_neg*model.gamma
    else:
        return -ll/float(len(uncens)+len(cens))


def conditional_distributions_loss(model, x, t, e, pdf_u, pdf_c, hr_loss=False, imbalance_loss=False, elbo=True, risk='1'):
        
    shape_weibull, scale_weibull, gates_weibull, shape_lognormal, scale_lognormal, logits_lognormal, attention_weights = model.forward(x)

    lossf_lognormal = []
    losss_lognormal = []

    hr_lognormal = []
    for g in range(model.k):

        mu = shape_lognormal[:, g]
        sigma = scale_lognormal[:, g]

        f = - sigma - 0.5*np.log(2*np.pi)
        f = f - torch.div((torch.log(t) - mu)**2, 2.*torch.exp(2*sigma))
        s = torch.div(torch.log(t) - mu, torch.exp(sigma)*np.sqrt(2))
        s = 0.5 - 0.5*torch.erf(s)
        s = torch.log(s)

        lossf_lognormal.append(f)
        losss_lognormal.append(s)

        # negative partial log likelihood
        hr_lognormal.append(f - s)

    losss_lognormal = torch.stack(losss_lognormal, dim=1)
    lossf_lognormal = torch.stack(lossf_lognormal, dim=1)
    hr_lognormal = torch.stack(hr_lognormal, dim=1)

    if elbo:
        lossg_lognormal = nn.Softmax(dim=1)(logits_lognormal)
        losss_lognormal = lossg_lognormal*losss_lognormal
        lossf_lognormal = lossg_lognormal*lossf_lognormal

        losss_lognormal = losss_lognormal.sum(dim=1)
        lossf_lognormal = lossf_lognormal.sum(dim=1)

        hr_lognormal = lossg_lognormal*hr_lognormal
        hr_lognormal = hr_lognormal.sum(dim=1)
    else:
        lossg_lognormal = nn.LogSoftmax(dim=1)(logits_lognormal)
        losss_lognormal = lossg_lognormal + losss_lognormal
        lossf_lognormal = lossg_lognormal + lossf_lognormal
        losss_lognormal = torch.logsumexp(losss_lognormal, dim=1)
        lossf_lognormal = torch.logsumexp(lossf_lognormal, dim=1)

    # Weibull distriubtion
    shapes_weibull, scales_weibull = shape_weibull.exp(), (-scale_weibull).exp()
    lossf_weibull, losss_weibull = [], []
    hr_weibull = []
    for idx in range(model.k):

        eta = shapes_weibull[:, idx]
        beta = scales_weibull[:, idx]
        
        log_s_weibull = - (torch.pow(t/beta, eta))
        log_f_weibull = torch.log(eta) - torch.log(beta) + ((eta-1)*(-torch.log(beta)+torch.log(t)))
        log_f_weibull = log_f_weibull + log_s_weibull

        lossf_weibull.append(log_f_weibull)
        losss_weibull.append(log_s_weibull)
        
        # negative partial log likelihood
        hr_weibull.append(torch.log(eta/beta*(t/beta)**(eta-1)))

    losss_weibull = torch.stack(losss_weibull, dim=1)
    lossf_weibull = torch.stack(lossf_weibull, dim=1)
    hr_weibull = torch.stack(hr_weibull, dim=1)

    if elbo:
        lossg_weibull = nn.Softmax(dim=1)(gates_weibull)
        losss_weibull = lossg_weibull*losss_weibull
        lossf_weibull = lossg_weibull*lossf_weibull
        losss_weibull = losss_weibull.sum(dim=1)
        lossf_weibull = lossf_weibull.sum(dim=1)
        hr_weibull = hr_weibull*lossg_weibull
        hr_weibull = hr_weibull.sum(dim=1)
    else:
        lossg_weibull = nn.LogSoftmax(dim=1)(gates_weibull)
        losss_weibull = lossg_weibull + losss_weibull
        lossf_weibull = lossg_weibull + lossf_weibull
        losss_weibull = torch.logsumexp(losss_weibull, dim=1)
        lossf_weibull = torch.logsumexp(lossf_weibull, dim=1)

    # Combine

    lossf , losss = torch.stack([lossf_lognormal, lossf_weibull], dim=1), torch.stack([losss_lognormal, losss_weibull], dim=1)
    weights  = nn.Softmax(dim=1)(attention_weights)
    #hr = torch.stack([hr_weibull, hr_lognormal], dim=1)
    hr = torch.stack([lossf_lognormal - losss_lognormal, lossf_weibull - losss_weibull], dim=1)
    hr = hr*weights
    hr = hr.sum(dim=1)
    loss_neg = PartialLogLikelihood()(hr, e)

    
    lossf = lossf*weights
    losss = losss*weights
    lossf = lossf.sum(dim=1)
    losss = losss.sum(dim=1)
    
    #
    if imbalance_loss:
        try:
            idx_time = t.int().cpu().detach().numpy()
            pdf_u_ = torch.tensor(pdf_u).cuda() 
            pdf_c_ = torch.tensor(pdf_c).cuda() 
            lossf = lossf * (1- pdf_u_[idx_time]) #.exp()
            losss = losss * (1- pdf_c_[idx_time]) #.exp()
        except:
            pass

    uncens = np.where(e.cpu().data.numpy() == int(risk))[0]
    cens = np.where(e.cpu().data.numpy() != int(risk))[0]
    ll = lossf[uncens].sum() + model.discount*losss[cens].sum()
    
    if hr_loss and e.sum() > 0:
        return -ll/float(len(uncens)+len(cens)) + loss_neg*model.gamma
    else:
        return -ll/float(len(uncens)+len(cens))
