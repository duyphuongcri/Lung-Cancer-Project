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

    print('Initialize network with %s' % init_type)
    net.apply(init_func)

def param_network(model):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    #print(model)
    print("The number of parameters: {}".format(num_params))

class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out, normalize="batchnorm"):
        super(single_conv,self).__init__()
        if "batchnorm" == normalize:
            self.conv = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm3d(ch_out),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1),
                nn.GroupNorm(num_groups=16,num_channels=ch_out),
                nn.ReLU(inplace=True)
            )            
    def forward(self,x):
        x = self.conv(x)
        return x

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out, normalize="batchnorm"):
        super(conv_block,self).__init__()
        if "batchnorm" == normalize:
            self.conv = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm3d(ch_out),
                nn.ReLU(inplace=True),
                nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm3d(ch_out),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1),
                nn.GroupNorm(num_groups=16,num_channels=ch_out),
                nn.ReLU(inplace=True),
                nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1),
                nn.GroupNorm(num_groups=16,num_channels=ch_out),
                nn.ReLU(inplace=True)
            )            
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out, normalize="batchnorm", scale_factor=2):
        super(up_conv,self).__init__()
        if "batchnorm" in normalize:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor),
                nn.Conv3d(ch_in,ch_out,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm3d(ch_out),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor),
                nn.Conv3d(ch_in,ch_out,kernel_size=3,stride=1,padding=1),
                nn.GroupNorm(num_groups=16,num_channels=ch_out),
                nn.ReLU(inplace=True)
            )            
    def forward(self,x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self, normalize="batchnorm", img_ch=1,output_ch=1):
        super(U_Net,self).__init__()
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=16, normalize=normalize )
        self.Conv2 = conv_block(ch_in=16,ch_out=32, normalize=normalize)
        self.Conv3 = conv_block(ch_in=32,ch_out=64, normalize=normalize)
        self.Conv4 = conv_block(ch_in=64,ch_out=128, normalize=normalize)
        self.Conv5 = conv_block(ch_in=128,ch_out=256, normalize=normalize)
        self.Conv6 = conv_block(ch_in=256,ch_out=512, normalize=normalize)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.Up7 = up_conv(ch_in=512,ch_out=512, normalize=normalize, scale_factor=(3,4,4))
        
        self.Up6 = up_conv(ch_in=512,ch_out=256, normalize=normalize)
        self.Up_conv6 = conv_block(ch_in=512, ch_out=256, normalize=normalize)

        self.Up5 = up_conv(ch_in=256,ch_out=128, normalize=normalize)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128, normalize=normalize)

        self.Up4 = up_conv(ch_in=128,ch_out=64, normalize=normalize)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64, normalize=normalize)
        
        self.Up3 = up_conv(ch_in=64,ch_out=32, normalize=normalize)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32, normalize=normalize)
        
        self.Up2 = up_conv(ch_in=32,ch_out=16, normalize=normalize)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16, normalize=normalize)

        self.Conv_1x1 = nn.Conv3d(16,output_ch,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        x6 = self.Maxpool(x5)
        x6 = self.Conv6(x6)

        x7 = self.avgpool(x6)
        d7 = self.Up7(x7)
        # decoding + concat path
        d6 = self.Up6(d7) #x6
        d6 = torch.cat((x5,d6),dim=1)
        d6 = self.Up_conv6(d6)

        d5 = self.Up5(d6)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1, x7