import os
import pandas as pd 
import numpy as np 
import torch 
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import argparse 
import misc, dataloader, utils, losses
import evaluation
from tqdm import tqdm 
from datetime import datetime 

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    # hyper-parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='epoch number')
    parser.add_argument('--lrate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='training batch size')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50,
                        help='every n epochs decay learning rate')
    parser.add_argument('--is_thop', type=bool, default=True,
                        help='whether calculate FLOPs/Params (Thop)')
    parser.add_argument('--path_train_image', type=str, default="D:\\LungCancer\\new dataset\\dataset\\train",
                        help='')
    parser.add_argument('--path_val_image', type=str, default="D:\\LungCancer\\new dataset\\dataset\\val",
                        help='')  
    parser.add_argument('--modelname', type=str, default="UNet",
                        help='Choose one of models: UNet, Attention_UNet, Attention_UNet_Trinh')    
    parser.add_argument('--modenor', type=str, default="groupnorm",
                        help='Choose one of methods: groupnorm, batchnorm') 

    arg = parser.parse_args()

    np.random.seed(10)
    torch.manual_seed(10)

    date = (datetime.now()).strftime("%Y-%m-%d")
    if not os.path.exists("./checkpoint/" + arg.modelname +"_bs_{}_{}_".format (arg.batch_size,arg.modenor)+ date):
        os.makedirs("./checkpoint/" + arg.modelname +"_bs_{}_{}_".format (arg.batch_size,arg.modenor)+ date)
    # Writer will output to ./runs/ directory by default
    if not os.path.exists("./info/" + arg.modelname +"_bs_{}_{}_".format (arg.batch_size,arg.modenor)+ date):
        os.makedirs("./info/" + arg.modelname +"_bs_{}_{}_".format (arg.batch_size,arg.modenor)+ date)
    writer = SummaryWriter("./info/" + arg.modelname +"_bs_{}_{}_".format (arg.batch_size,arg.modenor)+ date)

    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    if arg.modelname == "UNet":
        import models.Unet_3D as unet
        model = unet.U_Net(normalize=arg.modenor, img_ch=1,output_ch=1)
    elif arg.modelname == "Attention_UNet":
        import models.UNet_attention as unet
        model = unet.Att_UNet(normalize=arg.modenor,img_ch=1,output_ch=1)
    elif arg.modelname == "Attention_UNet_Trinh":
        import models.UNet_Atten_Trinh as unet
        model = unet.UNet_atten_Trinh(normalize=arg.modenor,img_ch=1,output_ch=1,ratio=16)

    model.to(device)
    model.apply(unet.init_weights)
    
    unet.param_network(model)
    optimizer = torch.optim.Adam(model.parameters(), arg.lrate)
    criterion = losses.SoftDiceLoss_v1()
    dice_metric = evaluation.DiceAccuracy_v1()

    for epoch in range(arg.epochs):
        #utils.adjust_lr(optimizer, arg.lrate, epoch, arg.decay_rate, arg.decay_epoch)
        loss_train, loss_test = 0, 0
        dice_train, dice_test = 0, 0
        n_train, n_test = 0, 0
        #### ------------Train-----------------------###
        model.train()
        for X, Y in tqdm(dataloader.load_CT_slices(arg.path_train_image, arg.batch_size, mode="train")):
            y_pred, _ = model(X.to(device))
            loss = criterion(y_pred, Y.to(device))
            dice_train += dice_metric(y_pred, Y.to(device)).item()*X.shape[0]
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()*X.shape[0]
            n_train += X.shape[0]

        # Valid
        model.eval()
        with torch.no_grad():
            for X, Y in tqdm(dataloader.load_CT_slices(arg.path_val_image, 1, mode="val")):
                y_pred, _ = model(X.to(device))
                loss = criterion(y_pred, Y.to(device))
                dice_test += dice_metric(y_pred, Y.to(device)).item()*X.shape[0]

                loss_test += loss.item()*X.shape[0]
                n_test += X.shape[0]
        
        # save log
        writer.add_scalar('Dice/test', dice_test/n_test, epoch)
        writer.add_scalar('Dice/train', dice_train/n_train, epoch)
        writer.add_scalar('Loss/test', loss_test/n_test, epoch)
        writer.add_scalar('Loss/train', loss_train/n_train, epoch)
        print("Epoch: {} | Loss/train: {:.04f} | Loss/test: {:.04f} | Dice/train: {:.04f} | Dice/test: {:.04f}".format(epoch, loss_train/n_train, loss_test/n_test,
                                                                                                                        dice_train/n_train, dice_test/n_test))
        #save checkpoint
        torch.save(model, "./checkpoint/" + arg.modelname +"_bs_{}_{}_".format (arg.batch_size,arg.modenor)+ date + "/model_{}.pt".format(epoch))

    writer.close()
        

        
