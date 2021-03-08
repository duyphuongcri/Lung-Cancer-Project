import os
import pandas as pd 
import numpy as np 
import torch 
import argparse 
import misc, dataloader, utils, losses
import dsm_models 
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime 
from sklearn.model_selection import StratifiedKFold, KFold
from pycox.evaluation import EvalSurv

def pretrain_dsm(premodel, optimizer, t_train, e_train, t_valid, e_valid, n_iter, thres):
    oldcost = float('inf')
    patience = 0
    costs = []
    for _ in tqdm(range(n_iter)):
        optimizer.zero_grad()
        loss = 0
        for r in range(1):
            loss += losses.weibull_loss(premodel, t_train, e_train, '1')
        loss.backward()
        optimizer.step()

        valid_loss = 0
        for r in range(1): 
            valid_loss += losses.weibull_loss(premodel, t_valid, e_valid, '1')
        valid_loss = valid_loss.detach().cpu().numpy()
        costs.append(valid_loss)
        #print(valid_loss)
        if np.abs(costs[-1] - oldcost) < thres:
            patience += 1
            if patience == 3:
                break
        oldcost = costs[-1]
    return premodel  

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        
        #m.bias.data.fill_(0.01)

if __name__=="__main__":
    
    for fold in range(5):
        print("Starting fold {}".format(fold))
        parser = argparse.ArgumentParser()
        # hyper-parameters
        parser.add_argument('--epochs', type=int, default=100,
                            help='epoch number')
        parser.add_argument('--lrate', type=float, default=0.01,
                            help='learning rate')
        parser.add_argument('--batch_size', type=int, default=8000,
                            help='training batch size')
        parser.add_argument('--clip', type=float, default=0.5,
                            help='gradient clipping margin')
        parser.add_argument('--decay_rate', type=float, default=0.1,
                            help='decay rate of learning rate')
        parser.add_argument('--decay_epoch', type=int, default=30,
                            help='every n epochs decay learning rate')
        parser.add_argument('--distribution', type=str, default="",
                            help='')
        parser.add_argument('--gamma', type=float, default=0.0,
                            help='')
        parser.add_argument('--hr_loss', type=bool, default=True,
                            help='')
        parser.add_argument('--imbalance_loss', type=bool, default=False,
                            help='')
        parser.add_argument('--path_clinical_train', type=str, default="./dataset/ct_pet_n_246/train_fold_{}.csv".format(fold),
                            help='')
        parser.add_argument('--path_clinical_val', type=str, default="./dataset/ct_pet_n_246/val_fold_{}.csv".format(fold),
                            help='')  
        parser.add_argument('--path_clinical_test', type=str, default="./dataset/ct_pet_n_246/test.csv",
                            help='')  
        parser.add_argument('--dataname', type=str, default="lungcancer",
                            help='Choose one of these options: lungcancer, Lung1-dataset, heart, support ')                                   
        arg = parser.parse_args()


        np.random.seed(20)# 25 20 25 20 20
        torch.manual_seed(20)


        date = (datetime.now()).strftime("%Y-%m-%d")
        if not os.path.exists("./checkpoint/" + arg.dataname + "/" + date + "_" + arg.distribution + "_hr_loss_{}_imbalanceloss_{}_bs_{}".format(arg.hr_loss,arg.imbalance_loss, arg.batch_size) + "_fold_{}".format(fold)):
            os.makedirs("./checkpoint/" + arg.dataname + "/" + date + "_" + arg.distribution + "_hr_loss_{}_imbalanceloss_{}_bs_{}".format(arg.hr_loss,arg.imbalance_loss, arg.batch_size)  + "_fold_{}".format(fold))
        # Writer will output to ./runs/ directory by default
        if not os.path.exists("./info/"  + arg.dataname + "/" + date + "_" + arg.distribution  + "_hr_loss_{}_imbalanceloss_{}_bs_{}".format(arg.hr_loss,arg.imbalance_loss, arg.batch_size)  + "_fold_{}".format(fold)):
            os.makedirs("./info/" + arg.dataname + "/" + date + "_" + arg.distribution + "_hr_loss_{}_imbalanceloss_{}_bs_{}".format(arg.hr_loss,arg.imbalance_loss, arg.batch_size)  + "_fold_{}".format(fold))
        writer = SummaryWriter("./info/"  + arg.dataname + "/" + date + "_" + arg.distribution + "_hr_loss_{}_imbalanceloss_{}_bs_{}".format(arg.hr_loss,arg.imbalance_loss, arg.batch_size)  + "_fold_{}".format(fold))


        device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

        if arg.distribution == "lognormal":
            model = dsm_models.DSM_LogNormal(k=1, layers=[100], gamma=arg.gamma, n_input=8)
        elif arg.distribution == "exponient":
            model = dsm_models.DSM_Weibull(k=1, layers=[100], gamma=arg.gamma, n_input=13)
            pretrain_model = dsm_models.DSM_Weibull(k=1, layers=[64, 64], gamma=arg.gamma, n_input=1)
        elif arg.distribution == "weibull":
            model = dsm_models.DSM_Weibull(k=1, layers=[64, 64], gamma=arg.gamma, n_input=13) # 5 7 11 18
            pretrain_model = dsm_models.DSM_Weibull(k=1, layers=[64, 64], gamma=arg.gamma, n_input=1)
        elif arg.distribution == "combine":
            model = dsm_models.Multiple_Distribution(k=1, layers=[50, 100])



        #     t_train = T /ratio
        #     e_train = E
        # for X, T, E in dataloader.load_data(arg.dataname, arg.path_clinical_val,  'test'):
        #     t_valid = T/ratio
        #     e_valid = E

        # optimizer = torch.optim.RMSprop(pretrain_model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

        # pretrained_model = pretrain_dsm(pretrain_model,  optimizer, t_train, e_train, t_valid, e_valid, n_iter=10000, thres=1e-4)

        # model.shape.data.fill_(float(pretrained_model.shape))
        # model.scale.data.fill_(float(pretrained_model.scale))

        
        ######################
        model.to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=arg.lrate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        #optimizer = torch.optim.Adam(model.parameters(), arg.lrate, weight_decay=0.0)
        min_loss = np.inf
        c_index_max = 0

        confidence = 0
        for epoch in range(arg.epochs):
            #utils.adjust_lr(optimizer, arg.lrate, epoch, arg.decay_rate, arg.decay_epoch)
            #### ------------Train-----------------------###
            loss_train, loss_test = 0, 0
            n_train, n_test = 0, 0
            model.train()
            for X, T, E in dataloader.load_data(arg.dataname, arg.path_clinical_train,  'Train'):
                ratio = float(int(T.max()/10 + int((T.max()/10) % 1 == 0)))
                T = T/ratio
                    
                pdf_u, pdf_c = misc.cal_pdf(T, E) 
                #print(pdf_u.sum(), pdf_c.sum())
                times = list(T.sort()[0].detach().numpy())
                times = list(dict.fromkeys(times))
                for idx in range(int(len(X)/ arg.batch_size)+1):
                    xb = X[arg.batch_size*idx: arg.batch_size*(idx+1)].to(device)
                    tb = T[arg.batch_size*idx: arg.batch_size*(idx+1)].to(device)
                    eb = E[arg.batch_size*idx: arg.batch_size*(idx+1)].to(device)
              
                    # # Sort
                    # _, indices = torch.sort(tb)
                    # indices = torch.flip(indices, dims=[0])
                    # eb = eb[indices]
                    # xb = xb[indices]
                    # tb = tb[indices]

              
                    optimizer.zero_grad()
                    if arg.distribution == "lognormal":
                        loss = losses.conditional_lognormal_loss(model, xb, tb, eb, pdf_u, pdf_c, hr_loss=arg.hr_loss, imbalance_loss=arg.imbalance_loss)
                    elif arg.distribution == "exponient":
                        loss = losses.conditional_exponient_loss(model, xb, tb, eb, pdf_u, pdf_c, hr_loss=arg.hr_loss, imbalance_loss=arg.imbalance_loss)
                    elif arg.distribution == "weibull":
                        loss = losses.conditional_weibull_loss(model, xb, tb, eb, pdf_u, pdf_c, hr_loss=arg.hr_loss, imbalance_loss=arg.imbalance_loss)
                    elif arg.distribution == "combine":
                        loss = losses.conditional_distributions_loss(model, xb, tb, eb,  pdf_u, pdf_c, hr_loss=arg.hr_loss, imbalance_loss=arg.imbalance_loss)
                    
                    loss.backward()
                    optimizer.step()
                    n_train+=1

                    loss_train += loss.item()

                    
                    if arg.distribution == "lognormal":
                        out_survival = utils.predict_survival_lognormal(model, X.to(device), times)
                    elif arg.distribution == "exponient":
                        out_survival = utils.predict_survival_exponient(model, X.to(device), times)
                    elif arg.distribution == "weibull":
                        out_survival = utils.predict_survival_weibull(model, X.to(device), times)
                    elif arg.distribution == "combine":
                        out_survival = utils.predict_survival_multiple_distributions(model, X.to(device), times)
                    surv = pd.DataFrame(out_survival, index=times)
                    durations_test, events_test = T.detach().numpy(), E.detach().numpy()
                    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
                    c_index_train = ev.concordance_td()

            ##### valid
            model.eval()
            batch_size = 8000
            with torch.no_grad():
                for X, T, E in dataloader.load_data(arg.dataname, arg.path_clinical_val, 'test'):
                    T = T/ratio

                    #pdf = misc.cal_pdf(T) 

                    if arg.distribution == "lognormal":
                        out_survival = utils.predict_survival_lognormal(model, X.to(device), times)
                    elif arg.distribution == "exponient":
                        out_survival = utils.predict_survival_exponient(model, X.to(device), times)
                    elif arg.distribution == "weibull":
                        out_survival = utils.predict_survival_weibull(model, X.to(device), times)
                    elif arg.distribution == "combine":
                        out_survival = utils.predict_survival_multiple_distributions(model, X.to(device), times)

                    
                    surv = pd.DataFrame(out_survival, index=times)
                    durations_test, events_test = T.detach().numpy(), E.detach().numpy()
                    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
                    c_index_val = ev.concordance_td()

                    for idx in range(int(len(X)/ batch_size)+1):
                        xb = X[batch_size*idx: batch_size*(idx+1)].to(device)
                        tb = T[batch_size*idx: batch_size*(idx+1)].to(device)
                        eb = E[batch_size*idx: batch_size*(idx+1)].to(device)

                        # # Sort
                        # _, indices = torch.sort(tb)
                        # indices = torch.flip(indices, dims=[0])
                        # eb = eb[indices]
                        # xb = xb[indices]
                        # tb = tb[indices]


                        if arg.distribution == "lognormal":
                            loss = losses.conditional_lognormal_loss(model, xb, tb, eb, pdf_u, pdf_c, hr_loss=arg.hr_loss, imbalance_loss=arg.imbalance_loss)
                        elif arg.distribution == "exponient":
                            loss = losses.conditional_exponient_loss(model, xb, tb, eb, pdf_u, pdf_c, hr_loss=arg.hr_loss, imbalance_loss=arg.imbalance_loss)
                        elif arg.distribution == "weibull":
                            loss = losses.conditional_weibull_loss(model, xb, tb, eb, pdf_u, pdf_c, hr_loss=arg.hr_loss, imbalance_loss=arg.imbalance_loss)
                        elif arg.distribution == "combine":
                            loss = losses.conditional_distributions_loss(model, xb, tb, eb,  pdf_u, pdf_c, hr_loss=arg.hr_loss, imbalance_loss=arg.imbalance_loss)

                        loss_test += loss.item()
                        n_test += 1
                # Test set
                for X, T, E in dataloader.load_data(arg.dataname, arg.path_clinical_test, 'test'):
                    T = T/ratio

                    if arg.distribution == "lognormal":
                        out_survival = utils.predict_survival_lognormal(model, X.to(device), times)
                    elif arg.distribution == "exponient":
                        out_survival = utils.predict_survival_exponient(model, X.to(device), times)
                    elif arg.distribution == "weibull":
                        out_survival = utils.predict_survival_weibull(model, X.to(device), times)
                    elif arg.distribution == "combine":
                        out_survival = utils.predict_survival_multiple_distributions(model, X.to(device), times)

                    surv = pd.DataFrame(out_survival, index=times)
                    durations_test, events_test = T.detach().numpy(), E.detach().numpy()
                    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
                    c_index_test = ev.concordance_td()  

            writer.add_scalar('Loss/train', loss_train/n_train, epoch)
            writer.add_scalar('Loss/test', loss_test/n_test, epoch)
            writer.add_scalar('C-index val', c_index_val, epoch)
            print("Epoch: {} | Loss/train: {:.04f} | Loss/test: {:.04f} | C-index Val: {:.04f} | C-index Test: {:.04f} | C-index Train: {:.04f}".format(epoch, loss_train/n_train, loss_test/n_test, c_index_val, c_index_test, c_index_train))

            # save model
            if loss_test < min_loss: # c_index_val> c_index_max: #  
                confidence = 0
                print('Saving model')
                #c_index_max = c_index_val
                min_loss = loss_test
                torch.save(model, "./checkpoint/"  + arg.dataname + "/" + date + "_" + arg.distribution + "_hr_loss_{}_imbalanceloss_{}_bs_{}".format(arg.hr_loss,arg.imbalance_loss, arg.batch_size)  + "_fold_{}".format(fold) + "/model_best.pt")
            else:
                confidence += 1
            torch.save(model, "./checkpoint/"  + arg.dataname + "/" + date + "_" + arg.distribution + "_hr_loss_{}_imbalanceloss_{}_bs_{}".format(arg.hr_loss,arg.imbalance_loss, arg.batch_size)  + "_fold_{}".format(fold) + "/model_{}.pt".format(epoch))
            
            if confidence >= 15: # Early stopping
                break
        writer.close()
        print()



        
