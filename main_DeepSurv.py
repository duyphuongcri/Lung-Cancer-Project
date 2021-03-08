import os
import pandas as pd 
import numpy as np 
import torch 
import argparse 
import misc, dataloader, utils, losses
from model import *
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime 
def convert_day_to_class(surv_time):
    surv_classses = torch.zeros(surv_time.shape)

    surv_classses[surv_time*7 >= 7*365] = 9
    surv_classses[surv_time*7 < 7*365] = 8
    surv_classses[surv_time*7 < 6*365] = 7
    surv_classses[surv_time*7 < 5*365] = 6
    surv_classses[surv_time*7 < 4*365] = 5
    surv_classses[surv_time*7 < 3*365] = 4
    surv_classses[surv_time*7 < 2*365] = 3
    surv_classses[surv_time*7 < 545] = 2
    surv_classses[surv_time*7 < 365] = 1
    surv_classses[surv_time*7 < 180] = 0

    return surv_classses

if __name__=="__main__":
    
    for fold in range (5):
        parser = argparse.ArgumentParser()
        # hyper-parameters
        parser.add_argument('--epochs', type=int, default=100,
                            help='epoch number')
        parser.add_argument('--lrate', type=float, default=1e-2,
                            help='learning rate')
        parser.add_argument('--batch_size', type=int, default=1,
                            help='training batch size')
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
        parser.add_argument('--path_clinical_test', type=str, default="./dataset/ct_pet_n_246/val_fold_{}.csv".format(fold),
                            help='')  
        parser.add_argument('--data_type', type=str, default="ct",
                            help='Choose one of these options: clinical, ct, pet, clinical_ct_pet')  
        parser.add_argument('--dataname', type=str, default="lungcancer",
                            help='Choose one of these options: lungcancer, Lung1-dataset ')                                     
        arg = parser.parse_args()

        np.random.seed(20)#10 10 11 15 15
        torch.manual_seed(20)

        date = (datetime.now()).strftime("%Y-%m-%d")
        if not os.path.exists("./checkpoint/" + date + "_" + arg.data_type + "_fold_{}".format(fold)):
            os.makedirs("./checkpoint/" + date + "_" + arg.data_type + "_fold_{}".format(fold))
        # Writer will output to ./runs/ directory by default
        if not os.path.exists("./info/" + date + "_" + arg.data_type + "_fold_{}".format(fold)):
            os.makedirs("./info/"+ date + "_" + arg.data_type + "_fold_{}".format(fold))
        writer = SummaryWriter("./info/" + date + "_" + arg.data_type + "_fold_{}".format(fold))


        device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
        if "clinical" == arg.data_type:
            model = DeepCPH_clinical(n_input=13)
        elif "clinical_ct_pet" == arg.data_type:
            model = DeepCPH()
        elif "ct" == arg.data_type or "pet" == arg.data_type:
            model = DeepCPH_image()

        model.to(device)
        param_network(model)
        optimizer = torch.optim.Adam(model.parameters(), arg.lrate, weight_decay=0.5)
        criterion = losses.PartialLogLikelihood()
        min_loss = np.inf
        for epoch in range(arg.epochs):
            #utils.adjust_lr(optimizer, arg.lrate, epoch, arg.decay_rate, arg.decay_epoch)
            #### ------------Train-----------------------###
            loss_train, loss_test = 0, 0
            n_train, n_test = 0, 0
            model.train()
            for X_CT, X_PET, X_Clinical, E, Y_true in dataloader.load_CT_PET_Clinical_data(arg.dataname, arg.path_image, arg.path_clinical_train, arg.batch_size, "train", arg.data_type):
                if "clinical" == arg.data_type:
                    log_hr = model(X_Clinical.to(device))
                elif "clinical_ct_pet" == arg.data_type:
                    log_hr = model(X_CT.to(device), X_PET.to(device), X_Clinical.to(device))
                elif "ct" == arg.data_type:
                    log_hr = model(X_CT.to(device))
                elif "pet" == arg.data_type:
                    log_hr = model( X_PET.to(device))
                #print(X_CT.shape)


                if len(torch.where(E)[0]) == 0:
                    continue

                # Sort
                _, indices = torch.sort(Y_true)
                indices = torch.flip(indices, dims=[0])
                E = E[indices]
                log_hr = log_hr[indices]

                loss = criterion(log_hr, E.to(device))
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
                n_train += 1

            # valid
            model.eval()
            with torch.no_grad():
                for X_CT, X_PET, X_Clinical, E, Y_true in dataloader.load_CT_PET_Clinical_data(arg.dataname, arg.path_image, arg.path_clinical_test, arg.batch_size, "test", arg.data_type):
                    if "clinical" == arg.data_type:
                        log_hr = model(X_Clinical.to(device))
                    elif "clinical_ct_pet" == arg.data_type:
                        log_hr = model(X_CT.to(device), X_PET.to(device), X_Clinical.to(device))
                    elif "ct" == arg.data_type:
                        log_hr = model(X_CT.to(device))
                    elif "pet" == arg.data_type:
                        log_hr = model( X_PET.to(device))

                #for X_Clinical, Y_true, E in dataloader.load_data(arg.dataname, arg.path_clinical_test, 'test'):
                    #log_hr = model(X_Clinical.to(device))

                    if len(torch.where(E)[0]) == 0:
                        continue
                    # Sort
                    _, indices = torch.sort(Y_true)
                    indices = torch.flip(indices, dims=[0])
                    E = E[indices]
                    log_hr = log_hr[indices]
                    

                    loss = criterion(log_hr, E.to(device))
                    loss_test += loss.item()
                    n_test += 1
            
            writer.add_scalar('Loss/train', loss_train/n_train, epoch)
            writer.add_scalar('Loss/test', loss_test/n_test, epoch)
            print("Epoch: {} | Loss/train: {:.04f} | Loss/test: {:.04f} ".format(epoch, loss_train/n_train, loss_test/n_test))
            # save model
            if min_loss > loss_test:
                print("save model")
                min_loss = loss_test
                torch.save(model, "./checkpoint/" + date + "_" + arg.data_type  + "_fold_{}".format(fold) + "/model_best.pt")
            torch.save(model, "./checkpoint/" + date + "_" + arg.data_type  + "_fold_{}".format(fold) + "/model_{}.pt".format(epoch))
        writer.close()



        
