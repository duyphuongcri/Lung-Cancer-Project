import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='xavier_uniform_', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier_normal_':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform_':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'kaiming_normal_':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'kaiming_uniform_':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1 or classname.find('GroupNorm') != -1:
            #init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    #print('Initialize network with %s' % init_type)
    net.apply(init_func)

def param_network(model):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    #print(model)
    print("The number of parameters: {}".format(num_params))

class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )
         
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out, scale_factor=2):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv3d(ch_in,ch_out,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        x = self.up(x)
        return x

def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes,  kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm3d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm3d(F_int))
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm3d(1),)
      
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        attention_w = self.sigmoid(psi)

        return x*attention_w

class ResAttUNet3D(nn.Module):
    """
    """
    def __init__(self, in_ch=1, out_ch=1):
        super(ResAttUNet3D, self).__init__()
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)

        self.Conv1 = nn.Sequential(
            single_conv(in_ch, 32),
            BasicBlock(32, 32, stride=1, downsample=None)
        )
        self.Conv2 = nn.Sequential(
            single_conv(32, 64),
            BasicBlock(64, 64, stride=1, downsample=None)
        )
        self.Conv3 = nn.Sequential(
            single_conv(64, 128),
            BasicBlock(128, 128, stride=1, downsample=None)
        )
        self.Conv4 = nn.Sequential(
            single_conv(128, 256),
            BasicBlock(256, 256, stride=1, downsample=None)
        )

        # decoder
        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv4 = nn.Sequential(
            single_conv(256, 128),
            BasicBlock(128, 128, stride=1, downsample=None)
        )
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv3 = nn.Sequential(
            single_conv(128, 64),
            BasicBlock(64, 64, stride=1, downsample=None)
        )
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block(F_g=32,F_l=32,F_int=16)
        self.Up_conv2 = nn.Sequential(
            single_conv(64, 32),
            BasicBlock(32, 32, stride=1, downsample=None)
        )

        self.Conv_1x1 = nn.Conv3d(32,out_ch,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # decoding + concat path

        d4 = self.Up4(x4)
        d4 = self.Att4(g=x3,x=d4)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.Att3(g=x2,x=d3)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.Att2(g=x1,x=d2)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv_1x1(d2)

        return out
       
# import numpy as np 
# device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

# model = ResAttUNet3D()
# model.to(device)

# x = np.ones((1, 1, 144, 128, 128))
# x = torch.Tensor(x).to(device)

# out = model(x)

# print(out.shape)