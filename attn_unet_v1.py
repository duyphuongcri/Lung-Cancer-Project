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
        elif classname.find('BatchNorm2d') != -1:
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
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
        
        
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class multi_res_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(multi_res_block,self).__init__()
        self.shortcut = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(ch_out)
        )
        self.conv3x3 = single_conv(ch_in,round(ch_out/6))
        self.conv5x5 = single_conv(round(ch_out/6),round(ch_out/3))
        self.conv7x7 = single_conv(round(ch_out/3),round(ch_out/2))
        
        self.batchnorm = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self,x):
        shortcut = self.shortcut(x)
        
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(conv3x3)
        conv7x7 = self.conv7x7(conv5x5)
        
        x_cat = torch.cat((conv3x3,conv5x5,conv7x7),dim=1) # 1 = 1/2 + 1/3 +1/6
        x_cat = self.batchnorm(x_cat)
        
        x = self.batchnorm(self.relu(shortcut+x_cat))
        
        return x
   
   
class res_path(nn.Module):
    def __init__(self,ch_in,ch_out,len):
        super(res_path,self).__init__()
        # t=0
        self.conv1x1 = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(ch_out)
        )
        self.conv3x3 = single_conv(ch_in,ch_out)
        
        # t>0
        self.conv1x1_ = nn.Sequential(nn.Conv2d(ch_out, ch_out, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(ch_out)
        )
        self.conv3x3_ = single_conv(ch_out,ch_out)
        
        
        self.len= len
        self.batchnorm = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        
        
    def forward(self,x):
        conv1x1 = self.conv1x1(x)
        conv3x3 = self.conv3x3(x)
        x = self.batchnorm(self.relu(conv1x1+conv3x3))    
        for i in range((self.len-1)):
            conv1x1 = self.conv1x1_(x)
            conv3x3 = self.conv3x3_(x)
            x = self.batchnorm(self.relu(conv1x1+conv3x3))
           
        return x
        

class MultiRes_U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(MultiRes_U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = multi_res_block(ch_in=img_ch,ch_out=round(32*1.8))
        self.Conv2 = multi_res_block(ch_in=round(32*1.8),ch_out=round(64*1.8))
        self.Conv3 = multi_res_block(ch_in=round(64*1.8),ch_out=round(128*1.8))
        self.Conv4 = multi_res_block(ch_in=round(128*1.8),ch_out=round(256*1.8))
        self.Conv5 = multi_res_block(ch_in=round(256*1.8),ch_out=round(512*1.8))

        self.Up5 = up_conv(ch_in=round(512*1.8),ch_out=256)
        self.res5 = res_path(ch_in=round(256*1.8),ch_out=256,len=1)
        self.Up_conv5 = multi_res_block(ch_in=256*2, ch_out=round(256*1.8))

        self.Up4 = up_conv(ch_in=round(256*1.8),ch_out=128)
        self.res4 = res_path(ch_in=round(128*1.8),ch_out=128,len=2)
        self.Up_conv4 = multi_res_block(ch_in=128*2, ch_out=round(128*1.8))
        
        self.Up3 = up_conv(ch_in=round(128*1.8),ch_out=64)
        self.res3 = res_path(ch_in=round(64*1.8),ch_out=64,len=3)
        self.Up_conv3 = multi_res_block(ch_in=64*2, ch_out=round(64*1.8))
        
        self.Up2 = up_conv(ch_in=round(64*1.8),ch_out=32)
        self.res2 = res_path(ch_in=round(32*1.8),ch_out=32,len=4)
        self.Up_conv2 = multi_res_block(ch_in=32*2, ch_out=round(32*1.8))

        self.Conv_1x1 = nn.Conv2d(round(32*1.8),output_ch,kernel_size=1,stride=1,padding=0)


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
        x4 = self.res5(x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.res4(x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.res3(x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.res2(x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
        
        
class MultiRes_U_Net_v1(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(MultiRes_U_Net_v1,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = multi_res_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = multi_res_block(ch_in=32,ch_out=64)
        self.Conv3 = multi_res_block(ch_in=64,ch_out=128)
        self.Conv4 = multi_res_block(ch_in=128,ch_out=256)
        self.Conv5 = multi_res_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.res5 = res_path(ch_in=256,len=1)
        self.Up_conv5 = multi_res_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.res4 = res_path(ch_in=128,len=2)
        self.Up_conv4 = multi_res_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.res3 = res_path(ch_in=64,len=3)
        self.Up_conv3 = multi_res_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.res2 = res_path(ch_in=32,len=4)
        self.Up_conv2 = multi_res_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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
        x4 = self.res5(x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.res4(x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.res3(x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.res2(x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class conv_block_nested(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output
        
        
class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, img_ch=3, output_ch=1):
        super(NestedUNet, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.Up = nn.Upsample(scale_factor=2)
        
        self.conv0_0 = conv_block_nested(img_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], output_ch, kernel_size=1,stride=1,padding=0)


    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat((x0_0, self.Up(x1_0)), dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat((x1_0, self.Up(x2_0)), dim=1))
        x0_2 = self.conv0_2(torch.cat((x0_0, x0_1, self.Up(x1_1)), dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat((x2_0, self.Up(x3_0)), dim=1))
        x1_2 = self.conv1_2(torch.cat((x1_0, x1_1, self.Up(x2_1)), dim=1))
        x0_3 = self.conv0_3(torch.cat((x0_0, x0_1, x0_2, self.Up(x1_2)), dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat((x3_0, self.Up(x4_0)), dim=1))
        x2_2 = self.conv2_2(torch.cat((x2_0, x2_1, self.Up(x3_1)), dim=1))
        x1_3 = self.conv1_3(torch.cat((x1_0, x1_1, x1_2, self.Up(x2_2)), dim=1))
        x0_4 = self.conv0_4(torch.cat((x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)), dim=1))

        output = self.final(x0_4)
        return output        
  
  
class NestedUNet_v1(nn.Module):
    """
    my implement
    """
    def __init__(self, img_ch=3, output_ch=1):
        super(NestedUNet_v1, self).__init__()

        #n1 = 32
        #filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.Up1_0 = up_conv(ch_in=64,ch_out=32)
        
        self.Up2_0 = up_conv(ch_in=128,ch_out=64)
        self.Up1_1 = up_conv(ch_in=64,ch_out=32)
        
        self.Up3_0 = up_conv(ch_in=256,ch_out=128)
        self.Up2_1 = up_conv(ch_in=128,ch_out=64)
        self.Up1_2 = up_conv(ch_in=64,ch_out=32)
        
        self.Up4_0 = up_conv(ch_in=512,ch_out=256)
        self.Up3_1 = up_conv(ch_in=256,ch_out=128)
        self.Up2_2 = up_conv(ch_in=128,ch_out=64)
        self.Up1_3 = up_conv(ch_in=64,ch_out=32)
        
        
        
        """
        self.conv0_0 = conv_block_nested(img_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        """
        self.conv0_0 = conv_block(ch_in=img_ch,ch_out=32)
        self.conv1_0 = conv_block(ch_in=32,ch_out=64)
        self.conv2_0 = conv_block(ch_in=64,ch_out=128)
        self.conv3_0 = conv_block(ch_in=128,ch_out=256)
        self.conv4_0 = conv_block(ch_in=256,ch_out=512)
        
        
        """
        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])
        """
        self.conv0_1 = conv_block(ch_in=32*2,ch_out=32)
        self.conv1_1 = conv_block(ch_in=64*2,ch_out=64)
        self.conv2_1 = conv_block(ch_in=128*2,ch_out=128)
        self.conv3_1 = conv_block(ch_in=256*2,ch_out=256)


        """
        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])
        """
        self.conv0_2 = conv_block(ch_in=32*3,ch_out=32)
        self.conv1_2 = conv_block(ch_in=64*3,ch_out=64)
        self.conv2_2 = conv_block(ch_in=128*3,ch_out=128)
       
        
        """
        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])
        """
        self.conv0_3 = conv_block(ch_in=32*4,ch_out=32)
        self.conv1_3 = conv_block(ch_in=64*4,ch_out=64)


        """
        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])
        """
        self.conv0_4 = conv_block(ch_in=32*5,ch_out=32)

        
        self.final = nn.Conv2d(32, output_ch, kernel_size=1,stride=1,padding=0)


    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.Maxpool(x0_0))
        x0_1 = self.conv0_1(torch.cat((x0_0, self.Up1_0(x1_0)), dim=1))

        x2_0 = self.conv2_0(self.Maxpool(x1_0))
        x1_1 = self.conv1_1(torch.cat((x1_0, self.Up2_0(x2_0)), dim=1))
        x0_2 = self.conv0_2(torch.cat((x0_0, x0_1, self.Up1_1(x1_1)), dim=1))

        x3_0 = self.conv3_0(self.Maxpool(x2_0))
        x2_1 = self.conv2_1(torch.cat((x2_0, self.Up3_0(x3_0)), dim=1))
        x1_2 = self.conv1_2(torch.cat((x1_0, x1_1, self.Up2_1(x2_1)), dim=1))
        x0_3 = self.conv0_3(torch.cat((x0_0, x0_1, x0_2, self.Up1_2(x1_2)), dim=1))

        x4_0 = self.conv4_0(self.Maxpool(x3_0))
        x3_1 = self.conv3_1(torch.cat((x3_0, self.Up4_0(x4_0)), dim=1))
        x2_2 = self.conv2_2(torch.cat((x2_0, x2_1, self.Up3_1(x3_1)), dim=1))
        x1_3 = self.conv1_3(torch.cat((x1_0, x1_1, x1_2, self.Up2_2(x2_2)), dim=1))
        x0_4 = self.conv0_4(torch.cat((x0_0, x0_1, x0_2, x0_3, self.Up1_3(x1_3)), dim=1))

        output = self.final(x0_4)
        return output  

  
class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
   
   
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class ChannelSELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):

        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):
    def __init__(self, num_channels):

        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, a, b))

        return output_tensor


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class Attention_block1(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block1,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0)

        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class Attention_block_v1(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block_v1,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return psi


class Attention_block_v2(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block_v2,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x + x*psi 


class Attention_block_v3(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block_v3,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(1)
        )
        
        self.relu = nn.ReLU(inplace=True)
        #self.up = nn.ConvTranspose2d( 1, 1, kernel_size=2, stride=2)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return psi


class Spatial_Attention_v0(nn.Module):
    def __init__(self,num_channels):
        super(Spatial_Attention_v0, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 1, kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_conv1 = self.conv1(x)
        psi = torch.cat([avg_out, max_out,x_conv1], dim=1)
        #psi = self.conv(x)

        return psi


class Spatial_Attention_v1(nn.Module):
    def __init__(self,num_channels):
        super(Spatial_Attention_v1, self).__init__()

        self.conv  = nn.Conv2d(3, 1, kernel_size=7, stride=1, padding=3)
        #self.conv  = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(num_channels, 1, kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_conv1 = self.conv1(x)
        x = torch.cat([avg_out, max_out,x_conv1], dim=1)
        psi = self.conv(x)

        return psi


class Spatial_Attention_v2(nn.Module):
    def __init__(self):
        super(Spatial_Attention_v2, self).__init__()

        self.conv  = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        psi = self.conv(x)

        return psi


class Spatial_Attention_v3(nn.Module):
    def __init__(self,num_channels):
        super(Spatial_Attention_v3, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 1, kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        psi = self.conv1(x)
        return psi


class Attention_block_v40(nn.Module):
    def __init__(self,F_g,F_l):
        super(Attention_block_v40,self).__init__()
        
        self.W_g = Spatial_Attention_v0(num_channels=F_g)
          
        self.W_x = Spatial_Attention_v0(num_channels=F_l)
        
        self.conv  = nn.Conv2d(3, 1, kernel_size=7, stride=1, padding=3)
        #self.conv  = nn.Conv2d(3, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,g,x):
    
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.sigmoid(self.conv(g1+x1))

        return psi


class Attention_block_v41(nn.Module):
    def __init__(self,F_g,F_l):
        super(Attention_block_v41,self).__init__()
        
        self.W_g = Spatial_Attention_v1(num_channels=F_g)
          
        self.W_x = Spatial_Attention_v1(num_channels=F_l)
     
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,g,x):
    
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.sigmoid(g1+x1)

        return psi


class Attention_block_v42(nn.Module):
    def __init__(self):
        super(Attention_block_v42,self).__init__()
        
        self.W_g = Spatial_Attention_v2()
          
        self.W_x = Spatial_Attention_v2()
     
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,g,x):
    
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.sigmoid(g1+x1)

        return psi


class Attention_block_v43(nn.Module):
    def __init__(self,F_g,F_l):
        super(Attention_block_v43,self).__init__()
             
        self.W_g = Spatial_Attention_v3(num_channels=F_g)
          
        self.W_x = Spatial_Attention_v3(num_channels=F_l)
     
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,g,x):
    
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.sigmoid(g1+x1)

        return psi


class Channel_Attention_v1(nn.Module):
    def __init__(self,num_channels, ratio=16):
        super(Channel_Attention_v1, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
       
       
        self.conv1 = nn.Conv2d(num_channels, num_channels//ratio, kernel_size=1)
        self.relu1 = nn.ReLU()
        #self.conv1 = nn.Conv2d(num_channels, num_channels//ratio, kernel_size=1,stride=1,padding=0)
    def forward(self, x):
        
        avg_out = self.relu1(self.conv1(self.avg_pool(x)))
        max_out = self.relu1(self.conv1(self.max_pool(x)))
        out = avg_out + max_out

        return out


class Channel_block_v41(nn.Module):
    def __init__(self,F_g,F_l,ratio=16):
        super(Channel_block_v41,self).__init__()
        
        self.W_g = Channel_Attention_v1(num_channels=F_g,ratio=ratio)
          
        self.W_x = Channel_Attention_v1(num_channels=F_l,ratio=ratio)
     
        self.conv1 = nn.Conv2d(F_l//ratio, F_l, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,g,x):
    
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.sigmoid(self.conv1(g1+x1))

        return psi


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(num_channels, num_channels // ratio, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(num_channels // ratio, num_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self,num_channels):
        super(SpatialAttention, self).__init__()

        self.conv  = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x)
        return self.sigmoid(out)


class SpatialAttention_v1(nn.Module):
    def __init__(self,num_channels):
        super(SpatialAttention_v1, self).__init__()

        self.conv  = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(num_channels, 1, kernel_size=1,stride=1,padding=0)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_conv1 = self.conv1(x)
        x = torch.cat([avg_out, max_out,x_conv1], dim=1)
        out = self.conv(x)
        return self.sigmoid(out)


class Attention_block_v5(nn.Module):
    def __init__(self, F_g,F_l, reduction_ratio=2):
        super(Attention_block_v5,self).__init__()
        
        num_channels = F_l
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        
        self.fc1 = nn.Linear(num_channels, num_channels_reduced)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self,g,x):
    
        batch_size_g, num_channels_g, H_g, W_g = g.size()
        batch_size_x, num_channels_x, H_x, W_x = x.size()

        squeeze_tensor_g = g.view(batch_size_g, num_channels_g, -1).mean(dim=2)
        squeeze_tensor_x = x.view(batch_size_x, num_channels_x, -1).mean(dim=2)
        
        squeeze_tensor = squeeze_tensor_g + squeeze_tensor_x
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        psi =  fc_out_2.view(a, b, 1, 1)
        
        return psi        


class Attention_block_v51(nn.Module):
    def __init__(self, F_g,F_l, reduction_ratio=2):
        super(Attention_block_v51,self).__init__()
        
        num_channels = F_g + F_l
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        
        self.fc1 = nn.Linear(num_channels, num_channels_reduced)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self,g,x):
    
        batch_size_g, num_channels_g, H_g, W_g = g.size()
        batch_size_x, num_channels_x, H_x, W_x = x.size()

        squeeze_tensor_g = g.view(batch_size_g, num_channels_g, -1).mean(dim=2)
        squeeze_tensor_x = x.view(batch_size_x, num_channels_x, -1).mean(dim=2)
        
        squeeze_tensor = torch.cat((squeeze_tensor_x,squeeze_tensor_g),dim=1)
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        
        a, b = squeeze_tensor.size()
        psi =  fc_out_2.view(a, b, 1, 1)
        
        return psi        


class Attention_block_v6(nn.Module):
    def __init__(self, F_g,F_l, reduction_ratio=2):
        super(Attention_block_v6, self).__init__()
        
        self.cSE = Attention_block_v5(F_g,F_l,reduction_ratio)
        self.sSE = Attention_block_v4(F_g,F_l)

    def forward(self, g, x):
        
        x1 = x * self.cSE(g,x)
        x2 = x * self.sSE(g,x)
        #psi = torch.max(x1,x2)
        psi = x1 + x2
        return psi


class Attention_block_v7(nn.Module):
    def __init__(self, F_g,F_l, reduction_ratio=2):
        super(Attention_block_v7, self).__init__()
        
        self.cSE = Attention_block_v5(F_g,F_l,reduction_ratio)
        self.sSE = Attention_block_v4(F_g,F_l)

    def forward(self, g, x):
        
        x1 = x * self.cSE(g,x)
        x2 = x * self.sSE(g,x)
        psi = torch.cat((x1,x2),dim=1)
        return psi


class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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

        return d1


class U_Net_deep(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net_deep,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        #self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
        self.Conv_1x1 = nn.Conv2d(480,output_ch,kernel_size=1,stride=1,padding=0)
        
        
        self.up_3 = nn.Upsample(scale_factor=2)
        self.up_4 = nn.Upsample(scale_factor=4)
        self.up_5 = nn.Upsample(scale_factor=8)

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


        d3_up = self.up_3(d3)
        d4_up = self.up_4(d4)
        d5_up = self.up_5(d5)
        
        d_concat = torch.cat((d2,d3_up,d4_up,d5_up),dim=1)
        
        d1 = self.Conv_1x1(d_concat)
        #d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=32,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=32,ch_out=64,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=128, ch_out=64,t=t)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=64, ch_out=32,t=t)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block(F_g=32,F_l=32,F_int=16)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
   
   
class AttU_Net_CBAM_v1(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net_CBAM_v1,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = SpatialAttention(num_channels=256)
        self.Cha5= ChannelAttention(num_channels=256,ratio=16)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = SpatialAttention(num_channels=128)
        self.Cha4= ChannelAttention(num_channels=128,ratio=16)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = SpatialAttention(num_channels=64)
        self.Cha3= ChannelAttention(num_channels=64,ratio=16)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = SpatialAttention(num_channels=32)
        self.Cha2= ChannelAttention(num_channels=32,ratio=16)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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
        x4 = x4 * self.Cha5(x4)
        x4 = x4 * self.Att5(x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = x3 * self.Cha4(x3)
        x3 = x3 * self.Att4(x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = x2 * self.Cha3(x2)
        x2 = x2 * self.Att3(x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = x1 * self.Cha2(x1)
        x1 = x1 * self.Att2(x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class AttU_Net_CBAM(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net_CBAM,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = SpatialAttention(num_channels=256)
        self.Cha5= ChannelAttention(num_channels=256,ratio=16)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = SpatialAttention(num_channels=128)
        self.Cha4= ChannelAttention(num_channels=128,ratio=16)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = SpatialAttention(num_channels=64)
        self.Cha3= ChannelAttention(num_channels=64,ratio=16)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = SpatialAttention(num_channels=32)
        self.Cha2= ChannelAttention(num_channels=32,ratio=16)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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
        a4 = self.Att5(x4)
        c4 = self.Cha5(x4)
        x4 = (x4*a4) * c4
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        a3 = self.Att4(x3)
        c3 = self.Cha4(x3)
        x3 = (x3*a3) * c3
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        a2 = self.Att3(x2)
        c2 = self.Cha3(x2)
        x2 = (x2*a2) * c2
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        a1 = self.Att2(x1)
        c1 = self.Cha2(x1)
        x1 = (x1*a1)* c1
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net_v411_encode(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net_v411_encode,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = SpatialAttention(num_channels=256)
        self.Cha5= ChannelAttention(num_channels=256,ratio=16)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = SpatialAttention(num_channels=128)
        self.Cha4= ChannelAttention(num_channels=128,ratio=16)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = SpatialAttention(num_channels=64)
        self.Cha3= ChannelAttention(num_channels=64,ratio=16)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = SpatialAttention(num_channels=32)
        self.Cha2= ChannelAttention(num_channels=32,ratio=16)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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
        a4 = self.Att5(x4)
        c4 = self.Cha5(x4)
        x4 = (x4*a4) * c4
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        a3 = self.Att4(x3)
        c3 = self.Cha4(x3)
        x3 = (x3*a3) * c3
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        a2 = self.Att3(x2)
        c2 = self.Cha3(x2)
        x2 = (x2*a2) * c2
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        a1 = self.Att2(x1)
        c1 = self.Cha2(x1)
        x1 = (x1*a1)* c1
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net_v411_decode(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net_v411_decode,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = SpatialAttention(num_channels=256)
        self.Cha5= ChannelAttention(num_channels=256,ratio=16)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = SpatialAttention(num_channels=128)
        self.Cha4= ChannelAttention(num_channels=128,ratio=16)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = SpatialAttention(num_channels=64)
        self.Cha3= ChannelAttention(num_channels=64,ratio=16)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = SpatialAttention(num_channels=32)
        self.Cha2= ChannelAttention(num_channels=32,ratio=16)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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
        a4 = self.Att5(d5)
        c4 = self.Cha5(d5)
        x4 = (x4*a4) * c4
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        a3 = self.Att4(d4)
        c3 = self.Cha4(d4)
        x3 = (x3*a3) * c3
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        a2 = self.Att3(d3)
        c2 = self.Cha3(d3)
        x2 = (x2*a2) * c2
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        a1 = self.Att2(d2)
        c1 = self.Cha2(d2)
        x1 = (x1*a1)* c1
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net_v1a(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net_v1a,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Spatial_Attention_v1(num_channels=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Spatial_Attention_v1(num_channels=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Spatial_Attention_v1(num_channels=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Spatial_Attention_v1(num_channels=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()

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
        a4 = self.sigmoid(self.Att5(x4))
        x4 = x4*a4
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        a3 = self.sigmoid(self.Att4(x3))
        x3 = x3*a3
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        a2 = self.sigmoid(self.Att3(x2))
        x2 = x2*a2
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        a1 = self.sigmoid(self.Att2(x1))
        x1 = x1*a1
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
        
        
class AttU_Net_v1b(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net_v1b,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Spatial_Attention_v1(num_channels=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Spatial_Attention_v1(num_channels=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Spatial_Attention_v1(num_channels=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Spatial_Attention_v1(num_channels=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()

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
        a4 = self.sigmoid(self.Att5(d5))
        x4 = x4*a4
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        a3 = self.sigmoid(self.Att4(d4))
        x3 = x3*a3
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        a2 = self.sigmoid(self.Att3(d3))
        x2 = x2*a2
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        a1 = self.sigmoid(self.Att2(d2))
        x1 = x1*a1
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net_v40(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net_v40,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_v40(F_g=256,F_l=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_v40(F_g=128,F_l=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_v40(F_g=64,F_l=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_v40(F_g=32,F_l=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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
        x4 = x4*a4
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        a3 = self.Att4(g=d4,x=x3)
        x3 = x3*a3
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        a2 = self.Att3(g=d3,x=x2)
        x2 = x2*a2
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        a1 = self.Att2(g=d2,x=x1)
        x1 = x1*a1
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net_v41(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net_v41,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_v41(F_g=256,F_l=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_v41(F_g=128,F_l=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_v41(F_g=64,F_l=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_v41(F_g=32,F_l=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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
        x4 = x4*a4
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        a3 = self.Att4(g=d4,x=x3)
        x3 = x3*a3
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        a2 = self.Att3(g=d3,x=x2)
        x2 = x2*a2
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        a1 = self.Att2(g=d2,x=x1)
        x1 = x1*a1
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net_v410(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,ratio=16):
        super(AttU_Net_v410,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        #self.Att5 = Attention_block_v41(F_g=256,F_l=256)
        self.Cha5= Channel_block_v41(F_g=256,F_l=256,ratio=ratio)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        #self.Att4 = Attention_block_v41(F_g=128,F_l=128)
        self.Cha4= Channel_block_v41(F_g=128,F_l=128,ratio=ratio)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        #self.Att3 = Attention_block_v41(F_g=64,F_l=64)
        self.Cha3= Channel_block_v41(F_g=64,F_l=64,ratio=ratio)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        #self.Att2 = Attention_block_v41(F_g=32,F_l=32)
        self.Cha2= Channel_block_v41(F_g=32,F_l=32,ratio=ratio)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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
        #a4 = self.Att5(g=d5,x=x4)
        c4 = self.Cha5(g=d5,x=x4)
        x4 = x4 * c4
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        #a3 = self.Att4(g=d4,x=x3)
        c3 = self.Cha4(g=d4,x=x3)
        x3 = x3 * c3
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        #a2 = self.Att3(g=d3,x=x2)
        c2 = self.Cha3(g=d3,x=x2)
        x2 = x2 * c2
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        #a1 = self.Att2(g=d2,x=x1)
        c1 = self.Cha2(g=d2,x=x1)
        x1 = x1 * c1
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net_v411(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,ratio=16):
        super(AttU_Net_v411,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_v41(F_g=256,F_l=256)
        self.Cha5= Channel_block_v41(F_g=256,F_l=256,ratio=ratio)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_v41(F_g=128,F_l=128)
        self.Cha4= Channel_block_v41(F_g=128,F_l=128,ratio=ratio)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_v41(F_g=64,F_l=64)
        self.Cha3= Channel_block_v41(F_g=64,F_l=64,ratio=ratio)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_v41(F_g=32,F_l=32)
        self.Cha2= Channel_block_v41(F_g=32,F_l=32,ratio=ratio)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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


class AttU_Net_v411_deep(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,ratio=16):
        super(AttU_Net_v411_deep,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_v41(F_g=256,F_l=256)
        self.Cha5= Channel_block_v41(F_g=256,F_l=256,ratio=ratio)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_v41(F_g=128,F_l=128)
        self.Cha4= Channel_block_v41(F_g=128,F_l=128,ratio=ratio)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_v41(F_g=64,F_l=64)
        self.Cha3= Channel_block_v41(F_g=64,F_l=64,ratio=ratio)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_v41(F_g=32,F_l=32)
        self.Cha2= Channel_block_v41(F_g=32,F_l=32,ratio=ratio)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_concat = conv_block(ch_in=480, ch_out=32)
        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)

        
        
        self.up_3 = nn.Upsample(scale_factor=2)
        self.up_4 = nn.Upsample(scale_factor=4)
        self.up_5 = nn.Upsample(scale_factor=8)


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


        d3_up = self.up_3(d3)
        d4_up = self.up_4(d4)
        d5_up = self.up_5(d5)
        
        d_concat = torch.cat((d2,d3_up,d4_up,d5_up),dim=1)
        d_concat = self.Conv_concat(d_concat)
        
        d1 = self.Conv_1x1(d_concat)
        #d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net_v411_deep1(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net_v411_deep1,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_v41(F_g=256,F_l=256)
        self.Cha5= Channel_block_v41(F_g=256,F_l=256,ratio=16)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_v41(F_g=128,F_l=128)
        self.Cha4= Channel_block_v41(F_g=128,F_l=128,ratio=16)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_v41(F_g=64,F_l=64)
        self.Cha3= Channel_block_v41(F_g=64,F_l=64,ratio=16)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_v41(F_g=32,F_l=32)
        self.Cha2= Channel_block_v41(F_g=32,F_l=32,ratio=16)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_concat = conv_block(ch_in=128, ch_out=32)
        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)

        
        
        self.up_3 = nn.Upsample(scale_factor=2)
        self.up_4 = nn.Upsample(scale_factor=4)
        self.up_5 = nn.Upsample(scale_factor=8)
        
        self.up_3_conv1x1 = nn.Conv2d(64,32,kernel_size=1,stride=1,padding=0)
        self.up_4_conv1x1 = nn.Conv2d(128,32,kernel_size=1,stride=1,padding=0)
        self.up_5_conv1x1 = nn.Conv2d(256,32,kernel_size=1,stride=1,padding=0)

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


        d3_up = self.up_3(self.up_3_conv1x1(d3))
        d4_up = self.up_4(self.up_4_conv1x1(d4))
        d5_up = self.up_5(self.up_5_conv1x1(d5))
        
        d_concat = torch.cat((d2,d3_up,d4_up,d5_up),dim=1)
        d_concat = self.Conv_concat(d_concat)
        
        d1 = self.Conv_1x1(d_concat)
        #d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net_v42(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net_v42,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_v42()
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_v42()
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_v42()
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_v42()
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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
        x4 = x4*a4
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        a3 = self.Att4(g=d4,x=x3)
        x3 = x3*a3
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        a2 = self.Att3(g=d3,x=x2)
        x2 = x2*a2
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        a1 = self.Att2(g=d2,x=x1)
        x1 = x1*a1
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net_v43(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net_v43,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_v43(F_g=256,F_l=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_v43(F_g=128,F_l=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_v43(F_g=64,F_l=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_v43(F_g=32,F_l=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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
        x4 = x4*a4
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        a3 = self.Att4(g=d4,x=x3)
        x3 = x3*a3
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        a2 = self.Att3(g=d3,x=x2)
        x2 = x2*a2
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        a1 = self.Att2(g=d2,x=x1)
        x1 = x1*a1
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net_v2(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net_v2,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_v5(F_g=256,F_l=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_v5(F_g=128,F_l=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_v5(F_g=64,F_l=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_v5(F_g=32,F_l=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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
        x4 = x4*a4
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        a3 = self.Att4(g=d4,x=x3)
        x3 = x3*a3
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        a2 = self.Att3(g=d3,x=x2)
        x2 = x2*a2
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        a1 = self.Att2(g=d2,x=x1)
        x1 = x1*a1
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net_v3(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net_v3,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_v6(F_g=256,F_l=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_v6(F_g=128,F_l=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_v6(F_g=64,F_l=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_v6(F_g=32,F_l=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net_v4(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net_v4,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_v7(F_g=256,F_l=256)
        self.Up_conv5 = conv_block(ch_in=768, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_v7(F_g=128,F_l=128)
        self.Up_conv4 = conv_block(ch_in=384, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_v7(F_g=64,F_l=64)
        self.Up_conv3 = conv_block(ch_in=192, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_v7(F_g=32,F_l=32)
        self.Up_conv2 = conv_block(ch_in=96, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class ResAttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(ResAttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_v1(F_g=256,F_l=256,F_int=128)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_v1(F_g=128,F_l=128,F_int=64)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_v1(F_g=64,F_l=64,F_int=32)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_v1(F_g=32,F_l=32,F_int=16)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
        

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
        #x4 = self.Att5(g=d5,x=x4)
        a4 = self.Att5(g=d5,x=x4)
        x4 = x4*a4
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        #x3 = self.Att4(g=d4,x=x3)
        a3 = self.Att4(g=d4,x=x3)
        a3 = (a3 + self.Upsample(a4))/2
        x3 = x3*a3
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        #x2 = self.Att3(g=d3,x=x2)
        a2 = self.Att3(g=d3,x=x2)
        a2 = (a2 + self.Upsample(a3))/2
        x2 = x2*a2
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        #x1 = self.Att2(g=d2,x=x1)
        a1 = self.Att2(g=d2,x=x1)
        a1 = (a1 + self.Upsample(a2))/2
        x1 = x1*a1
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class ResAttU_Net_v1(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(ResAttU_Net_v1,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_v1(F_g=256,F_l=256,F_int=128)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_v1(F_g=128,F_l=128,F_int=64)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_v1(F_g=64,F_l=64,F_int=32)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_v1(F_g=32,F_l=32,F_int=16)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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
        #x4 = self.Att5(g=d5,x=x4)
        a4 = self.Att5(g=d5,x=x4)
        x4 = x4 + x4*a4
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        #x3 = self.Att4(g=d4,x=x3)
        a3 = self.Att4(g=d4,x=x3)
        a3 = (a3 + self.Upsample(a4))
        x3 = x3 + x3*a3
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        #x2 = self.Att3(g=d3,x=x2)
        a2 = self.Att3(g=d3,x=x2)
        a2 = (a2 + self.Upsample(a3))
        x2 = x2 + x2*a2
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        #x1 = self.Att2(g=d2,x=x1)
        a1 = self.Att2(g=d2,x=x1)
        a1 = (a1 + self.Upsample(a2))
        x1 = x1 + x1*a1
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class ResAttU_Net_v2(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(ResAttU_Net_v2,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_v3(F_g=256,F_l=256,F_int=128)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_v3(F_g=128,F_l=128,F_int=64)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_v3(F_g=64,F_l=64,F_int=32)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_v3(F_g=32,F_l=32,F_int=16)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
        
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.ConvTranspose2d( 1, 1, kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()


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
        #x4 = self.Att5(g=d5,x=x4)
        a4 = self.Att5(g=d5,x=x4)
        a41 = self.sigmoid(a4)
        x4 = x4 + x4 * a41
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        #x3 = self.Att4(g=d4,x=x3)
        a3 = self.Att4(g=d4,x=x3)
        a31 = self.sigmoid(a3 + self.up(a4))
        x3 = x3 + x3 * a31
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        #x2 = self.Att3(g=d3,x=x2)
        a2 = self.Att3(g=d3,x=x2)
        a21 = self.sigmoid(a2 + self.up(a3))
        x2 = x2 + x2 * a21
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        #x1 = self.Att2(g=d2,x=x1)
        a1 = self.Att2(g=d2,x=x1)
        a11 = self.sigmoid(a1 + self.up(a2))
        x1 = x1 + x1*a11
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class ResAttU_Net_v4(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(ResAttU_Net_v4,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Conv4_1 = nn.Conv2d(256, 256, kernel_size=1,stride=1,padding=0)
        self.Conv3_1 = nn.Conv2d(128, 128, kernel_size=1,stride=1,padding=0)
        self.Conv2_1 = nn.Conv2d(64,   64, kernel_size=1,stride=1,padding=0)
        self.Conv1_1 = nn.Conv2d(32,   32, kernel_size=1,stride=1,padding=0)
        
        
        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_v3(F_g=256,F_l=256,F_int=128)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_v3(F_g=128,F_l=128,F_int=64)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_v3(F_g=64,F_l=64,F_int=32)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_v3(F_g=32,F_l=32,F_int=16)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
        
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.ConvTranspose2d( 1, 1, kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()


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
        #x4 = self.Att5(g=d5,x=x4)
        a4 = self.Att5(g=d5,x=x4)
        a41 = self.sigmoid(a4)
        x4 = self.Conv4_1(x4) + x4 * a41
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        #x3 = self.Att4(g=d4,x=x3)
        a3 = self.Att4(g=d4,x=x3)
        a31 = self.sigmoid(a3 + self.up(a4))
        x3 = self.Conv3_1(x3) + x3 * a31
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        #x2 = self.Att3(g=d3,x=x2)
        a2 = self.Att3(g=d3,x=x2)
        a21 = self.sigmoid(a2 + self.up(a3))
        x2 = self.Conv2_1(x2) + x2 * a21
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        #x1 = self.Att2(g=d2,x=x1)
        a1 = self.Att2(g=d2,x=x1)
        a11 = self.sigmoid(a1 + self.up(a2))
        x1 = self.Conv1_1(x1) + x1*a11
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=32,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=32,ch_out=64,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN5 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN4 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN3 = RRCNN_block(ch_in=128, ch_out=64,t=t)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = Attention_block(F_g=32,F_l=32,F_int=16)
        self.Up_RRCNN2 = RRCNN_block(ch_in=64, ch_out=32,t=t)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
        
                
class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        return out


class GAU_v1(nn.Module):
    def __init__(self, F_g, F_l):
        super(GAU_v1, self).__init__()

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_l, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(F_l)
            )
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_l, kernel_size=1),
            nn.BatchNorm2d(F_l)
            )


        self.relu = nn.ReLU(inplace=True)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, g, x):

        avg_g = self.avg_pool(g)
        fms_g = self.W_g(avg_g)
        fms_g = self.relu(fms_g)
        x1 = self.W_x(x)

        out = x1 * fms_g

        return out


class U_Net_GAU(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net_GAU,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = GAU_v1(F_g=256,F_l=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = GAU_v1(F_g=128,F_l=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = GAU_v1(F_g=64,F_l=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Att2 = GAU_v1(F_g=32,F_l=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


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
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
        
