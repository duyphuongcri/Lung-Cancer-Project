import os
import pandas as pd 
import numpy as np 
import torch 
import argparse 
import misc, dataloader, utils, losses
from model_classification import *
from tqdm import tqdm 

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    # hyper-parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='epoch number')
    parser.add_argument('--lrate', type=float, default=1e-6,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='training batch size')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50,
                        help='every n epochs decay learning rate')
    parser.add_argument('--is_thop', type=bool, default=True,
                        help='whether calculate FLOPs/Params (Thop)')
    parser.add_argument('--path_image', type=str, default="./dataset/total_12_folders",
                        help='')
    parser.add_argument('--path_clinical', type=str, default="./dataset/test5.csv",
                        help='')
    # parser.add_argument('--data_type', type=str, default="./dataset/test5.csv",
    #                     help='')                        
    arg = parser.parse_args()

    np.random.seed(10)
    torch.manual_seed(10)

    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    model = DeepCPH()
    model.to(device)
    param_network(model)
    optimizer = torch.optim.Adam(model.parameters(), arg.lrate, weight_decay=0.99)
    criterion = losses.NewLoss()
    #criterion = losses.PartialLogLikelihood()
    for epoch in range(arg.epochs):
        #utils.adjust_lr(optimizer, arg.lrate, epoch, arg.decay_rate, arg.decay_epoch)
        #### ------------Train-----------------------###
        loss_train = 0
        model.train()
        flag = True
        event, Y_pred_all = [], [] # initilize
        sum_h = 0
        n_fail_indicator = 0
        
        loss_mini_batch = 0
        loss = 0
        for X_CT, X_PET, X_Clinical, E, Y_true, Y_true_mean in tqdm(dataloader.load_CT_PET_Clinical_data(arg.path_image, arg.path_clinical, arg.batch_size)):
            y_pre = model(X_CT.to(device), X_PET.to(device))
        #     sum_h += np.exp(y_pre.cpu().detach().numpy())

        #     if len(torch.where(E)[0]) == 0:
        #         continue
        #     else:
        #         n_fail_indicator += 1
                
        #     if n_fail_indicator %500 == 0:
        #         print(y_pre.item(), sum_h)
        #     loss += criterion(y_pre, torch.Tensor(sum_h).to(device)) 
        #     #loss.backward(retain_graph=True)
        #     loss_train += loss.item()
        #     if (n_fail_indicator+1) % 24 == 0:
        #         optimizer.zero_grad()
        #         loss.backward(retain_graph=True)
        #         optimizer.step()
        #         loss=0
        # print(epoch, loss_train)   
        #torch.save(model, "./checkpoint/model_{}.pt".format(epoch))
        


        
