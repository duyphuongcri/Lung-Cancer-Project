import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math, copy 
from einops import rearrange

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

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)

        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.num_attention_heads = 16
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(0)
        self.proj_dropout = nn.Dropout(0)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        #print(x.shape, new_x_shape, x.size()[:-1])
        x = x.view(*new_x_shape)
        #print(x.shape, x.permute(0, 2, 1, 3).shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output 

class ExternelAttention(nn.Module):
    def __init__(self, hiddensize, num_memory_units=64, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(hiddensize / self.num_heads)

        self.query = nn.Linear(hiddensize, hiddensize)
        self.memory_key = nn.Linear(hiddensize // self.num_heads, num_memory_units)
        self.memory_value = nn.Linear(num_memory_units, hiddensize// self.num_heads)

        self.proj = nn.Linear(hiddensize, hiddensize)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):

        query_fea = self.query(x)
        query_layer = self.transpose_for_scores(query_fea)
        attn = self.memory_key(query_layer)
        attn = attn.softmax(dim=2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        out = self.memory_value(attn)
        out = rearrange(out, 'b h n c -> b n (h c)')
        out = self.proj(out)

        return out

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size):
        super(TransformerLayer, self).__init__()

        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ExterLayer = ExternelAttention(hidden_size)#multihead self-attention
        self.mlp = MLP(hidden_size)

    def forward(self, x):
        """
        x shape: (batchsize, n_patches, hidden_size)
        """
        h = x
        x = self.layernorm(x)
        x = self.ExterLayer(x)
        x = x + h

        h = x
        x = self.layernorm(x)
        x = self.mlp(x)
        x = x + h

        return x  

class SequentialTransformerLayers(nn.Module):
    def __init__(self, hidden_size, num_trans_layers):
        super(SequentialTransformerLayers, self).__init__()
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.layers = nn.ModuleList()
        for _ in range(num_trans_layers):
            layer = TransformerLayer(hidden_size)
            self.layers.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layers:
            hidden_states = layer_block(hidden_states) ##(batchsize, n_patches, hidden_size)

        encoded = self.layernorm(hidden_states)

        return encoded

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self,patch_size,resolution, in_channels, hidden_size):
        super(Embeddings, self).__init__()
        """
        patch_embeddings: to tokenize image patches
        hidden_size: D
        """
        n_patches = (resolution[0] // patch_size[0]) * (resolution[1] // patch_size[1]) * (resolution[2] // patch_size[2])
        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                       out_channels=hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.patch_embeddings(x)  # (batchsize, hidden_size, patch size h, patch size w)
        x = x.flatten(2)  # (batchsize, hidden_size, patch size h * patch size w)
        x = x.transpose(-1, -2)  #(batchsize, n_patches, hidden_size)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class ConvTransEncoder(nn.Module):
    def __init__(self, in_channels, hidden_size, patch_size, resolution,  num_trans_layers):
        super(ConvTransEncoder, self).__init__()

        self.embeddings = Embeddings(patch_size, resolution, in_channels, hidden_size)
        self.encoder = SequentialTransformerLayers(hidden_size, num_trans_layers)
        
    def forward(self, x):
        embedding_output = self.embeddings(x)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        shape = encoded.shape
        encoded = encoded.permute(0, 2, 1)
        n = math.ceil((x.shape[2]*x.shape[3]*x.shape[4]/shape[1])**(1./3))
        encoded = encoded.view(shape[0], shape[2], x.shape[2]//n, x.shape[3]//n, x.shape[4]//n)
        return encoded

class MAESegResNet3D(nn.Module):
    """
    This model was implemented from https://arxiv.org/pdf/1606.06650.pdf
    """
    def __init__(self, in_ch=1, out_ch=1):
        super(MAESegResNet3D, self).__init__()
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
        # Exterformer

        self.Att1 = ConvTransEncoder(32, 32, patch_size=(8, 8, 8), resolution=(144, 128, 128), num_trans_layers=1)
        self.Att2 = ConvTransEncoder(64, 64, patch_size=(4, 4, 4), resolution=(72, 64, 64), num_trans_layers=1)
        self.Att3 = ConvTransEncoder(128, 128, patch_size=(2, 2, 2), resolution=(36, 32, 32), num_trans_layers=1)
        self.Att4 = ConvTransEncoder(256, 256, patch_size=(1, 1, 1), resolution=(18, 16, 16), num_trans_layers=1)
        self.conv = single_conv(480, 256)
        # decoder
        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv4 = nn.Sequential(
            single_conv(256, 128),
            BasicBlock(128, 128, stride=1, downsample=None)
        )
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv3 = nn.Sequential(
            single_conv(128, 64),
            BasicBlock(64, 64, stride=1, downsample=None)
        )
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
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

        ##
        a4 = self.Att4(x4)
        a3 = self.Att3(x3)
        a2 = self.Att2(x2)
        a1 = self.Att1(x1)
        #print(a1.shape, a2.shape, a3.shape, a4.shape)
        d4 = torch.cat([a1, a2, a3, a4], dim=1)
        d4 = self.conv(d4)

        # decoding + concat path
        d4 = self.Up4(d4)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv_1x1(d2)

        return out
       
