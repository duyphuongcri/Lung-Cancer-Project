import os
import pandas as pd 
import numpy as np 
import torch 
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import argparse 
import misc, dataloader, utils, evaluation
import evaluation
from tqdm import tqdm 
from datetime import datetime 
import cv2 
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    # hyper-parameters
    parser.add_argument('--model_path', type=str, default="./checkpoint/Attention_UNet_Trinh_bs_2_groupnorm_2020-12-17/model_92.pt",
                        help='path to model')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='training batch size')
    parser.add_argument('--is_thop', type=bool, default=True,
                        help='whether calculate FLOPs/Params (Thop)')
    parser.add_argument('--path_test_image', type=str, default="D:\\LungCancer\\2020-lung-cancer\\New folder\\data", #D:\\LungCancer\\new dataset\\dataset\\train
                        help='D:\\LungCancer\\2020-lung-cancer\\New folder\\data,  D:\\LungCancer\\new dataset\\dataset\\val')  
           
    arg = parser.parse_args()

    np.random.seed(10)
    torch.manual_seed(10)


    if "UNet" in arg.model_path:
        import models.Unet_3D as unet
    elif "Attention_UNet" in arg.model_path:
        import models.UNet_attention as unet
    elif "Attention_UNet_Trinh" in arg.model_path:
        import models.UNet_Atten_Trinh as unet

    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    model = torch.load(arg.model_path)
    model.to(device)

    unet.param_network(model)
    dice_metric = evaluation.DiceAccuracy_v1()

    # Valid
    model.eval()
    dice_score = 0
    n_test = 0
    with torch.no_grad():
        for X, Y in tqdm(dataloader.load_CT_slices(arg.path_test_image, arg.batch_size, mode="test")):
            y_pred, latent_fea = model(X.to(device))
            # latent_fea = latent_fea.detach().cpu().numpy()
            # latent_fea = latent_fea.reshape(-1,1)
            # kmedoids = KMedoids(n_clusters=5, random_state=0).fit(latent_fea)
            # print(kmedoids.labels_)
            dice_score += dice_metric(y_pred, Y.to(device))*X.shape[0]
            n_test += X.shape[0]


            X = torch.squeeze(X).detach().cpu().numpy()
            Y = torch.squeeze(Y).detach().cpu().numpy()
            y_pred = torch.squeeze(torch.sigmoid(y_pred)).detach().cpu().numpy()
            #print(len(np.where(y_pred>0.5)[0]), len(np.where(Y>0.5)[0]))
            fig, ax = plt.subplots(1,1)
            plots = []
            for i in range(X.shape[0]):
                # _, img_bw = cv2.threshold((Y[i]*255.).astype(np.uint8), 150, 255, cv2.THRESH_BINARY)
                # _, contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
                # out = cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
                #print(img.shape, img.min(), img.max())
                plots.append(np.hstack((X[i], y_pred[i], Y[i])))
                #plots.append(X[i])

            y = np.dstack(plots)
            tracker = misc.IndexTracker(ax, y)
            fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
            plt.show()
    print("Dice_score: {:.04f}".format(dice_score/n_test), n_test)

        

        
