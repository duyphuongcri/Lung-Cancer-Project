import os
import pandas as pd 
import numpy as np 
import torch 
import argparse 
import misc, dataloader, utils, losses
from model import *
import dsm_models
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime 


if __name__=="__main__":
    
    for fold in range (5):
        parser = argparse.ArgumentParser()
        # hyper-parameters
        parser.add_argument('--epochs', type=int, default=100,
                            help='epoch number')
        parser.add_argument('--lrate', type=float, default=0.01,
                            help='learning rate')
        parser.add_argument('--batch_size', type=int, default=8000,
                            help='training batch size')
        parser.add_argument('--distribution', type=str, default="weibull",
                            help='')
        parser.add_argument('--gamma', type=float, default=0.1,
                            help='')
        parser.add_argument('--hr_loss', type=bool, default=False,
                            help='')
        parser.add_argument('--imbalance_loss', type=bool, default=False,
                            help='')
        parser.add_argument('--clip', type=float, default=0.5,
                            help='gradient clipping margin')
        parser.add_argument('--decay_rate', type=float, default=0.1,
                            help='decay rate of learning rate')
        parser.add_argument('--decay_epoch', type=int, default=40,
                            help='every n epochs decay learning rate')
        parser.add_argument('--is_thop', type=bool, default=True,
                            help='whether calculate FLOPs/Params (Thop)')
        parser.add_argument('--path_image', type=str, default="./dataset/ct_pet_n_246",
                            help='')
        parser.add_argument('--path_clinical_train', type=str, default="./dataset/ct_pet_n_246/train_fold_{}.csv".format(fold),
                            help='')
        parser.add_argument('--path_clinical_val', type=str, default="./dataset/ct_pet_n_246/val_fold_{}.csv".format(fold),
                            help='')  
        parser.add_argument('--data_type', type=str, default="clinical",
                            help='Choose one of these options: clinical, ct, pet, clinical_ct_pet')  
        parser.add_argument('--dataname', type=str, default="lungcancer",
                            help='Choose one of these options: lungcancer, Lung1-dataset ')                                     
        arg = parser.parse_args()

        np.random.seed(20)#10 10 11 15 15
        torch.manual_seed(20)

        date = (datetime.now()).strftime("%Y-%m-%d")
        if not os.path.exists("./checkpoint/"  + arg.dataname + "/" + date + "_" + arg.data_type + "_" + arg.distribution + "_hr_loss_{}_imbalanceloss_{}_bs_{}".format(arg.hr_loss,arg.imbalance_loss, arg.batch_size)  + "_fold_{}".format(fold)):
            os.makedirs("./checkpoint/"  + arg.dataname + "/" + date + "_" + arg.data_type + "_" + arg.distribution + "_hr_loss_{}_imbalanceloss_{}_bs_{}".format(arg.hr_loss,arg.imbalance_loss, arg.batch_size)  + "_fold_{}".format(fold))
        # Writer will output to ./runs/ directory by default
        if not os.path.exists("./info/"  + arg.dataname + "/" + date + "_" + arg.data_type + "_" + arg.distribution + "_hr_loss_{}_imbalanceloss_{}_bs_{}".format(arg.hr_loss,arg.imbalance_loss, arg.batch_size)  + "_fold_{}".format(fold)):
            os.makedirs("./info/"  + arg.dataname + "/" + date + "_" + arg.data_type + "_" + arg.distribution + "_hr_loss_{}_imbalanceloss_{}_bs_{}".format(arg.hr_loss,arg.imbalance_loss, arg.batch_size)  + "_fold_{}".format(fold))
        writer = SummaryWriter("./info/"  + arg.dataname + "/" + date + "_" + arg.data_type + "_" + arg.distribution + "_hr_loss_{}_imbalanceloss_{}_bs_{}".format(arg.hr_loss,arg.imbalance_loss, arg.batch_size)  + "_fold_{}".format(fold))


        device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
        if "clinical" == arg.data_type:
            model = dsm_models.DSM_Weibull(k=1, layers=[64, 64], gamma=arg.gamma, n_input=13) # 5 7 11 18
        elif "ct" == arg.data_type or "pet" == arg.data_type:
            model = dsm_models.DSM_Weibull_image(k=1, layers=[64, 64], gamma=arg.gamma, n_input=13) # 5 7 11 18

        model.to(device)
        param_network(model)
        optimizer = torch.optim.Adam(model.parameters(), arg.lrate, weight_decay=0.5)
        criterion = losses.PartialLogLikelihood()
        min_loss = np.inf
        confidence = 0
        for epoch in range(arg.epochs):
            #utils.adjust_lr(optimizer, arg.lrate, epoch, arg.decay_rate, arg.decay_epoch)
            #### ------------Train-----------------------###
            loss_train, loss_val = 0, 0
            n_train, n_val = 0, 0
            model.train()
            for X_CT, X_PET, X_Clinical, E, T, T_max in dataloader.load_CT_PET_Clinical_data(arg.dataname, arg.path_image, arg.path_clinical_train, arg.batch_size, "train", arg.data_type):
                ratio = float(int(T_max/10 + int((T_max/10) % 1 == 0)))
                T = T/ratio
                    
                pdf_u, pdf_c = misc.cal_pdf(T, E) 

                optimizer.zero_grad()

                loss = losses.conditional_weibull_loss(arg.data_type, model, X_CT, X_PET, X_Clinical, T.to(device), E.to(device), pdf_u, pdf_c, hr_loss=arg.hr_loss, imbalance_loss=arg.imbalance_loss,  device=device)

                loss.backward()
                optimizer.step()
                n_train+=1

                loss_train += loss.item()

                    
            # valid
            model.eval()
            with torch.no_grad():
                for X_CT, X_PET, X_Clinical, E, T, T_max in dataloader.load_CT_PET_Clinical_data(arg.dataname, arg.path_image, arg.path_clinical_val, arg.batch_size, "test", arg.data_type):
                    T = T/ratio
                    loss = losses.conditional_weibull_loss(arg.data_type, model, X_CT, X_PET, X_Clinical, T.to(device), E.to(device), pdf_u, pdf_c, hr_loss=arg.hr_loss, imbalance_loss=arg.imbalance_loss,  device=device)
                    n_val += 1
                    loss_val += loss.item()
            
            writer.add_scalar('Loss/train', loss_train/n_train, epoch)
            writer.add_scalar('Loss/val', loss_val/n_val, epoch)
            print("Epoch: {} | Loss/train: {:.04f} | Loss/val: {:.04f} ".format(epoch, loss_train/n_train, loss_val/n_val))
            # save model
            if min_loss > loss_val:
                confidence = 0
                print("save model")
                min_loss = loss_val
                torch.save(model, "./checkpoint/"  + arg.dataname + "/" + date + "_" + arg.data_type + "_" + arg.distribution + "_hr_loss_{}_imbalanceloss_{}_bs_{}".format(arg.hr_loss,arg.imbalance_loss, arg.batch_size)  + "_fold_{}".format(fold) + "/model_best.pt")
            else:
                confidence += 1
            torch.save(model, "./checkpoint/"  + arg.dataname + "/" + date + "_" + arg.data_type + "_" + arg.distribution + "_hr_loss_{}_imbalanceloss_{}_bs_{}".format(arg.hr_loss,arg.imbalance_loss, arg.batch_size)  + "_fold_{}".format(fold) + "/model_{}.pt".format(epoch))
            
            # if confidence >= 15: # Early stopping
            #     break
        writer.close()
        print()



        
