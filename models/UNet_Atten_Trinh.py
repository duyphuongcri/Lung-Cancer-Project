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
    def __init__(self,ch_in,ch_out, normalize="batchnorm", num_groups=16):
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
                nn.GroupNorm(num_groups=num_groups,num_channels=ch_out),
                nn.ReLU(inplace=True)
            )            
    def forward(self,x):
        x = self.conv(x)
        return x

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out, normalize="batchnorm", num_groups=16):
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
                nn.GroupNorm(num_groups=num_groups,num_channels=ch_out),
                nn.ReLU(inplace=True),
                nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1),
                nn.GroupNorm(num_groups=num_groups,num_channels=ch_out),
                nn.ReLU(inplace=True)
            )            
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out, normalize="batchnorm", num_groups=16, scale_factor=2):
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
                nn.GroupNorm(num_groups=num_groups,num_channels=ch_out),
                nn.ReLU(inplace=True)
            )            
    def forward(self,x):
        x = self.up(x)
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

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l):
        super(Attention_block,self).__init__()
        
        self.W_g = Spatial_Attention(num_channels=F_g)
        self.W_x = Spatial_Attention(num_channels=F_l)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,g,x):
    
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.sigmoid(g1+x1)

        return psi

class Spatial_Attention(nn.Module):
    def __init__(self,num_channels):
        super(Spatial_Attention, self).__init__()

        self.conv  = nn.Conv3d(3, 1, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv3d(num_channels, 1, kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_conv1 = self.conv1(x)
        x = torch.cat([avg_out, max_out,x_conv1], dim=1)
        psi = self.conv(x)

        return psi

class Channel_block(nn.Module):
    def __init__(self,F_g,F_l,ratio=16):
        super(Channel_block,self).__init__()
        
        self.W_g = Channel_Attention(num_channels=F_g,ratio=ratio)
        self.W_x = Channel_Attention(num_channels=F_l,ratio=ratio)
        self.conv1 = nn.Conv3d(F_l//ratio, F_l, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,g,x):
    
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.sigmoid(self.conv1(g1+x1))

        return psi

class Channel_Attention(nn.Module):
    def __init__(self,num_channels, ratio=16):
        super(Channel_Attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.conv1 = nn.Conv3d(num_channels, num_channels//ratio, kernel_size=1)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        
        avg_out = self.relu1(self.conv1(self.avg_pool(x)))
        max_out = self.relu1(self.conv1(self.max_pool(x)))
        out = avg_out + max_out

        return out

class UNet_atten_Trinh(nn.Module):
    def __init__(self, normalize="batchnorm",img_ch=1,output_ch=1,ratio=16):
        super(UNet_atten_Trinh,self).__init__()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32, normalize=normalize)
        self.Conv2 = conv_block(ch_in=32,ch_out=64, normalize=normalize)
        self.Conv3 = conv_block(ch_in=64,ch_out=128, normalize=normalize)
        self.Conv4 = conv_block(ch_in=128,ch_out=256, normalize=normalize)
        self.Conv5 = conv_block(ch_in=256,ch_out=512, normalize=normalize)

        self.Up5 = up_conv(ch_in=512,ch_out=256, normalize=normalize)
        self.Att5 = Attention_block(F_g=256,F_l=256)
        self.Cha5= Channel_block(F_g=256,F_l=256,ratio=ratio)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256, normalize=normalize)

        self.Up4 = up_conv(ch_in=256,ch_out=128, normalize=normalize)
        self.Att4 = Attention_block(F_g=128,F_l=128)
        self.Cha4= Channel_block(F_g=128,F_l=128,ratio=ratio)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128, normalize=normalize)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64, normalize=normalize)
        self.Att3 = Attention_block(F_g=64,F_l=64)
        self.Cha3= Channel_block(F_g=64,F_l=64,ratio=ratio)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64, normalize=normalize)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32, normalize=normalize)
        self.Att2 = Attention_block(F_g=32,F_l=32)
        self.Cha2= Channel_block(F_g=32,F_l=32,ratio=ratio)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32, normalize=normalize)

        self.Conv_1x1 = nn.Conv3d(32, output_ch, kernel_size=1, stride=1, padding=0)

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


        # decoding + concat path
        d5 = self.Up5(x5)
        a4 = self.Att5(g=d5,x=x4)
        c4 = self.Cha5(g=d5,x=x4)
        x4 = (x4*a4) * c4
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        a3 = self.Att4(g=d4,x=x3)
        c3 = self.Cha4(g=d4,x=x3)
        x3 = (x3*a3) * c3
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        a2 = self.Att3(g=d3,x=x2)
        c2 = self.Cha3(g=d3,x=x2)
        x2 = (x2*a2) * c2
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        a1 = self.Att2(g=d2,x=x1)
        c1 = self.Cha2(g=d2,x=x1)
        x1 = (x1*a1)* c1
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        
        return d1
