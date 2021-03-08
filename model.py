import torch
import torch.nn as nn
import numpy as np 
import models.resnet as backbone
import math
def param_network(model):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    #print(model)
    print("The number of parameters: {}".format(num_params))


class DeepCPH(nn.Module):
    def __init__(self):
        super(DeepCPH, self).__init__()
        self.ct_model  = backbone.generate_model(34, n_input_channels=1)
        self.pet_model = backbone.generate_model(34, n_input_channels=1)

        # self.histology = nn.Linear(4, 1, bias=True)
        # self.gender  = nn.Linear(2, 1, bias=True) 
  
        self.linear1 = nn.Linear(1024, 128, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=128)

        self.linear2 = nn.Linear(128, 32, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=32)

        self.linear3 = nn.Linear(39, 1, bias=False)

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2, inplace=True)

        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, X_CT, X_PET, X_clinical):
        out = torch.cat((self.ct_model(X_CT), self.pet_model(X_PET)), dim=1)
        # out = self.relu(out)
        # out = self.dropout(out)

        out = self.linear1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # history = self.histology(X_clinical[:, :4])
        # gender = self.gender(X_clinical[:, 4:6])
        # out = torch.cat((history, gender, X_clinical[:, 6:], out), dim=1)

        out = torch.cat(( X_clinical[:, 6:], out), dim=1)

        out = self.linear3(out)

        return out

class DeepCPH_clinical(nn.Module):
    def __init__(self, n_input):
        super(DeepCPH_clinical, self).__init__()
        # self.histology = nn.Linear(4, 1, bias=True)
        # self.gender  = nn.Linear(2, 1, bias=True) 
  
        self.linear1 = nn.Linear(n_input, 64, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.linear2 = nn.Linear(64, 16, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=16)
        self.linear3 = nn.Linear(16, 1, bias=False)

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1, inplace=True)

        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        # history = self.histology(X_clinical[:, :4])
        # gender = self.gender(X_clinical[:, 4:6])
        # out = torch.cat((history, gender, X_clinical[:, 6:]), dim=1)

        out = X #[:, 6:]

        out = self.linear1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear3(out)

        return out

class DeepCPH_image(nn.Module):
    def __init__(self):
        super(DeepCPH_image, self).__init__()
        self.img_model  = backbone.generate_model(34, n_input_channels=1)

        self.linear1 = nn.Linear(512, 128, bias=True)
        self.linear2 = nn.Linear(128, 32, bias=True)
        self.linear3 = nn.Linear(32, 1, bias=False)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2, inplace=True)

        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        out = self.img_model(X)

        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear3(out)

        return out      


class DeepCPH_v2(nn.Module):
    def __init__(self):
        super(DeepCPH_v2, self).__init__()
        self.ct_model  = backbone.generate_model(34, n_input_channels=1)
        self.pet_model = backbone.generate_model(34, n_input_channels=1)

        self.histology = nn.Linear(4, 1, bias=True)
        self.gender  = nn.Linear(2, 1, bias=True) 
  
        self.linear1 = nn.Linear(1024, 128, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=128)

        self.linear2 = nn.Linear(128, 32, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=32)

        self.linear3 = nn.Linear(41, 1, bias=False)

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2, inplace=True)

        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, X_CT, X_PET, X_clinical):
        out = torch.cat((self.ct_model(X_CT), self.pet_model(X_PET)), dim=1)

        out = self.linear1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        history = self.histology(X_clinical[:, :4])
        gender = self.gender(X_clinical[:, 4:6])
        out = torch.cat((history, gender, X_clinical[:, 6:], out), dim=1)

        #out = torch.cat(( X_clinical[:, 6:], out), dim=1)

        out = self.linear3(out)

        return out

# class Distribution_network(nn.Module):
#     def __init__(self):
#         super(Distribution_network, self).__init__()
#         self.ct_model  = backbone.generate_model(18, n_input_channels=1)
#         self.pet_model = backbone.generate_model(18, n_input_channels=1)
  
#         self.linear1 = nn.Linear(807, 1024, bias=True)
#         self.linear2 = nn.Linear(1024, 256, bias=True)
#         self.linear3 = nn.Linear(256, 32, bias=True)
#         self.linear4 = nn.Linear(32, 1, bias=True)
#         self.relu    = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.2, inplace=True)

#         self.reset_parameters()
    
#     def reset_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight,
#                                         mode='fan_out',
#                                         nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm3d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, X_CT, X_PET, X_clinical):
#         ct  = self.ct_model(X_CT)
#         pet = self.pet_model(X_PET)
#         out =  torch.cat((ct, pet, X_clinical), dim=1)

#         out = self.linear1(out)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.linear2(out)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.linear3(out)
#         out = self.relu(out)
#         out = self.dropout(out)   
#         out = self.linear4(out)
#         return out