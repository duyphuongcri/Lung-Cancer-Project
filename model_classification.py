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
        self.ct_model  = backbone.generate_model(18, n_input_channels=1)
        self.pet_model = backbone.generate_model(18, n_input_channels=1)


        self.linear1 = nn.Linear(1024, 256, bias=True)
        self.linear2 = nn.Linear(256, 32, bias=True)
        self.linear3 = nn.Linear(32, 1, bias=True)
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

    def forward(self, X_CT, X_PET):
        out = torch.cat((self.ct_model(X_CT), self.pet_model(X_PET)), dim=1)
        print(self.ct_model(X_CT).shape)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear3(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out

# class DeepCPH(nn.Module):
#     def __init__(self):
#         super(DeepCPH, self).__init__()
#         self.ct_model  = backbone.generate_model(18, n_input_channels=1)
#         self.pet_model = backbone.generate_model(18, n_input_channels=1)
#         self.histology = nn.Linear(4, 1, bias=True)
#         self.O_stage = nn.Linear(4, 1, bias=True)
#         self.T_stage = nn.Linear(4, 1, bias=True)
#         self.N_stage = nn.Linear(4, 1, bias=True)
#         self.M_stage = nn.Linear(2, 1, bias=True)
#         self.gender  = nn.Linear(2, 1, bias=True) 
#         self.linear1 = nn.Linear(808, 1024, bias=True)
#         self.linear2 = nn.Linear(1024, 128, bias=True)
#         self.linear3 = nn.Linear(128, 1, bias=True)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, X_CT, X_PET, X_clinical):
#         ct_fea  = self.ct_model(X_CT)
#         pet_fea = self.pet_model(X_PET)
#         X_clinical = X_clinical.float()
#         history = self.histology(X_clinical[:, :4].float())
#         o_stage = self.O_stage(X_clinical[:, 4:8])
#         t_stage = self.T_stage(X_clinical[:, 8:12])
#         n_stage = self.N_stage(X_clinical[:, 12:16])
#         m_stage = self.M_stage(X_clinical[:, 16:18])
#         gender = self.gender(X_clinical[:, 18:20])
#         cli_fea = torch.cat((history, o_stage, t_stage, n_stage, m_stage, gender, X_clinical[:, -2:]), dim=1)

#         fea_combined = torch.cat((ct_fea, pet_fea, cli_fea), dim=1)
#         out = self.linear1(fea_combined)
#         out = self.relu(out)
#         out = self.linear2(out)
#         out = self.relu(out)
#         out = self.linear3(out)
   
#         return out

class Distribution_network(nn.Module):
    def __init__(self):
        super(Distribution_network, self).__init__()
        self.ct_model  = backbone.generate_model(18, n_input_channels=1)
        self.pet_model = backbone.generate_model(18, n_input_channels=1)
  
        self.linear1 = nn.Linear(807, 1024, bias=True)
        self.linear2 = nn.Linear(1024, 256, bias=True)
        self.linear3 = nn.Linear(256, 32, bias=True)
        self.linear4 = nn.Linear(32, 1, bias=True)
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
        ct  = self.ct_model(X_CT)
        pet = self.pet_model(X_PET)
        out =  torch.cat((ct, pet, X_clinical), dim=1)

        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.dropout(out)   
        out = self.linear4(out)
        return out