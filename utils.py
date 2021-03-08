import torch
import torch.nn as nn
import numpy as np

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def l2_regularisation(m):
    """https://github.com/stefanknegt/Probabilistic-Unet-Pytorch/blob/master/utils.py"""
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def exponient_cdf(model, x, t_horizon, risk='1'):
    
    squish = nn.LogSoftmax(dim=1)

    shape, scale, logits = model(x)
    
    shapes, scales = shape.exp(), (-scale).exp()
    
    logits = squish(logits)
    
    k_ = shape
    b_ = scale

    t_horz = torch.tensor(t_horizon)#.double()
    t_horz = t_horz.repeat(shape.shape[0], 1)

    cdfs = []
    pdfs = []
    for j in range(len(t_horizon)):

        t = t_horz[:, j]
        lcdfs = []
        lpdfs = []
        for idx in range(model.k):

            eta = shapes[:, idx].cpu()
            beta = scales[:, idx].cpu()

            log_s = -t*eta
            log_f = torch.log(eta) + log_s
            
            lcdfs.append(log_s)
            lpdfs.append(log_f.exp())  
            
        lpdfs = torch.stack(lpdfs, dim=1)
        lcdfs = torch.stack(lcdfs, dim=1)
        lpdfs = lpdfs*logits.exp().cpu()
        lcdfs = lcdfs + logits.cpu()
        lpdfs = torch.sum(lpdfs, dim=1)
        lcdfs = torch.logsumexp(lcdfs, dim=1)
        pdfs.append(lpdfs.detach().numpy())
        cdfs.append(lcdfs.detach().numpy())

    return cdfs, pdfs, shape, scale

def predict_survival_exponient(model, x, t, risk=1):
    cdfs, pdfs, shape, scale = exponient_cdf(model, x, t, risk=str(risk))
    return np.exp(np.array(cdfs)) #, np.exp(np.array(pdfs)), shape, scale


def weibull_cdf(model, X_CT, X_PET, X_Clinical, t_horizon, data_type, device, risk='1'):
    
    squish = nn.LogSoftmax(dim=1)

    if "ct" == data_type:
        shape, scale, logits = model(X_CT.to(device))
    elif "pet" == data_type:
        shape, scale, logits = model(X_PET.to(device))
    elif "clinical" == data_type:
        shape, scale, logits = model(X_Clinical.to(device))   

    shapes, scales = shape.exp(), (-scale).exp()
    
    logits = squish(logits)
    
    k_ = shape
    b_ = scale

    t_horz = torch.tensor(t_horizon)#.double()
    t_horz = t_horz.repeat(shape.shape[0], 1)

    cdfs = []
    pdfs = []
    for j in range(len(t_horizon)):

        t = t_horz[:, j]
        lcdfs = []
        lpdfs = []
        for idx in range(model.k):

            eta = shapes[:, idx].cpu()
            beta = scales[:, idx].cpu()

            log_s = - (torch.pow(t/beta, eta))
            log_f = torch.log(eta) - torch.log(beta) + ((eta-1)*(-torch.log(beta)+torch.log(t)))
            log_f = log_f + log_s
            
            lcdfs.append(log_s)
            lpdfs.append(log_f.exp())  
            
        lpdfs = torch.stack(lpdfs, dim=1)
        lcdfs = torch.stack(lcdfs, dim=1)
        lpdfs = lpdfs*logits.exp().cpu()
        lcdfs = lcdfs + logits.cpu()
        lpdfs = torch.sum(lpdfs, dim=1)
        lcdfs = torch.logsumexp(lcdfs, dim=1)
        pdfs.append(lpdfs.detach().numpy())
        cdfs.append(lcdfs.detach().numpy())

    return cdfs, pdfs, shape, scale

def predict_survival_weibull(model, X_CT, X_PET, X_Clinical, t, data_type, device,risk=1):
    cdfs, pdfs, shape, scale = weibull_cdf(model, X_CT, X_PET, X_Clinical, t, data_type, device, risk=str(risk))
    return np.exp(np.array(cdfs)) #, np.exp(np.array(pdfs)), shape, scale


def lognormal_cdf(model, x, t_horizon, risk='1'):

    squish = nn.LogSoftmax(dim=1)

    shape, scale, logits = model.forward(x)
    logits = squish(logits)

    k_ = shape
    b_ = scale

    t_horz = torch.tensor(t_horizon).double()
    t_horz = t_horz.repeat(shape.shape[0], 1)

    cdfs = []

    for j in range(len(t_horizon)):

        t = t_horz[:, j]
        lcdfs = []

        for g in range(model.k):

            mu = k_[:, g].cpu()
            sigma = b_[:, g].cpu()

            s = torch.div(torch.log(t) - mu, torch.exp(sigma)*np.sqrt(2))
            s = 0.5 - 0.5*torch.erf(s)
            s = torch.log(s)
            lcdfs.append(s)

        lcdfs = torch.stack(lcdfs, dim=1)
        lcdfs = lcdfs+logits.cpu()
        lcdfs = torch.logsumexp(lcdfs, dim=1)
        cdfs.append(lcdfs.detach().numpy())

    return cdfs

def predict_survival_lognormal(model, x, t, risk=1):
    cdfs = lognormal_cdf(model, x, t, risk=str(risk))
    return np.exp(np.array(cdfs))

def lognormal_weibull_cdf(model, x, t_horizon, risk='1'):
    
    shape_weibull, scale_weibull, gates_weibull, shape_lognormal, scale_lognormal, logits_lognormal, attention_weights = model.forward(x)
    
    shapes, scales = shape_weibull.exp(), (-scale_weibull).exp()
    
    
    # Log normal Distribution
    squish = nn.LogSoftmax(dim=1)
    attetion_w = squish(attention_weights)
    logits_lognormal = squish(logits_lognormal)
    gates_weibull = squish(gates_weibull)
    
    t_horz = torch.tensor(t_horizon).double()
    t_horz = t_horz.repeat(shape_weibull.shape[0], 1)

    cdfs = []
    
    for j in range(len(t_horizon)):

        t = t_horz[:, j]
        lcdfs_lognormal, log_weibull = [], []

        for idx in range(model.k):

            mu = shape_lognormal[:, idx].cpu()
            sigma = scale_lognormal[:, idx].cpu()

            s = torch.div(torch.log(t) - mu, torch.exp(sigma)*np.sqrt(2))
            s = 0.5 - 0.5*torch.erf(s)
            s = torch.log(s)
            lcdfs_lognormal.append(s)

            
            eta = shapes[:, idx].cpu()
            beta = scales[:, idx].cpu()
            log_s = - (torch.pow(t/beta, eta))
            log_weibull.append(log_s) 
            
        lcdfs_lognormal = torch.stack(lcdfs_lognormal, dim=1)
        lcdfs_lognormal = lcdfs_lognormal + logits_lognormal.cpu()
        lcdfs_lognormal = torch.logsumexp(lcdfs_lognormal, dim=1)
        #lcdfs_lognormal.append(lcdfs_lognormal.detach().numpy()) 
        
        
        log_weibull = torch.stack(log_weibull, dim=1)
        log_weibull = log_weibull + gates_weibull.cpu()
        log_weibull = torch.logsumexp(log_weibull, dim=1)
        
        ##
        lcdfs = torch.stack([lcdfs_lognormal, log_weibull], dim=1)
        lcdfs = lcdfs + attetion_w.cpu()
        lcdfs = torch.logsumexp(lcdfs, dim=1).exp()
        cdfs.append(lcdfs.detach().numpy())
    cdfs = np.array(cdfs)
    return cdfs

def predict_survival_multiple_distributions(model, x, t, risk=1):
    cdfs = lognormal_weibull_cdf(model, x, t, risk=str(risk))
    return cdfs
