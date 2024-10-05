import torch.nn as nn
import torch.nn.functional as F
import torch
import glob
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from augment import *
from math import floor
from PIL import Image
from dataset import img_generator, data_generate
from utils import get_Pkey, get_Pstring, str_to_paras, filter_files, try_convert2numeric
from grad_cam import GradCAM, show_cam_on_image, center_crop_img
from torchvision import transforms


augmentations = {}
augmentations["rcs"] = RCS # 随机颜色偏移
augmentations["acs"] = ACS # 自适应颜色偏移
augmentations["mixchannel"] = MixChannel
augmentations["multiviewfilter"] = MultiViewFilter
augmentations["vcs"] = VCS # 变分颜色偏移
augmentations["dvcs"] = DCVCS
augmentations['pcacs'] = PCACS

INPUT_CHANNEL = 3

Models = {}
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

class ConvTrans(nn.Module):
    def __init__(self, in_c, out_c, stride, kernel_size=1):
        super(ConvTrans, self).__init__()
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        
    def forward(self, x):
        out = self.trans(x)
        return out

class DeepCAPTCHA(nn.Module):
    CHW = {
        'T':{'c':INPUT_CHANNEL,'h':64,'w':192}, 
        '0':{'c':INPUT_CHANNEL,'h':64,'w':192}, 
        '1':{'c':32,'h':32,'w':96}, 
        '2':{'c':48,'h':16,'w':48}, 
        '3':{'c':64,'h':8,'w':24}
    }
    def __init__(self, PARAS=None, num_of_classes=26, CBAM=False, ratio=8, STN=False, TPS=False, EQU=False, 
        WEIGHTS=False, FILTER=False, pool_type=['avg', 'max'], DEEPCAPTCHA=True, RES=False, FPN=False, LOSS=False, 
        AUG=False, KERNEL=None):
        super(DeepCAPTCHA, self).__init__()
        self.num_of_classes = num_of_classes
        self.CBAM = CBAM
        self.ratio = ratio
        self.STN = STN
        self.TPS = TPS
        self.FILTER = FILTER
        self.RES = RES
        self.FPN = FPN
        self.PARAS = PARAS
        self.EQU = EQU
        self.WEIGHTS = WEIGHTS
        self.AUG = AUG
        self.DEEPCAPTCHA = DEEPCAPTCHA
        self.KERNEL = KERNEL
        self.num_of_classes = num_of_classes
        self.in_channel = INPUT_CHANNEL

        myConv2d = eval(PARAS.KERNEL) if KERNEL else nn.Conv2d
        # 1*64*192-> 32*32*96
        self.conv1 = myConv2d(INPUT_CHANNEL, 32, kernel_size=5, padding='same')
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # 32*32*96 -> 48*16*48
        self.conv2 = myConv2d(32, 48, kernel_size=5, padding='same')
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # 48*16*48 -> 64*8*24
        self.conv3 = myConv2d(48, 64, kernel_size=5, padding='same')
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.dropout_linear = nn.Dropout(0.3)
        self.linear1 = nn.Linear(512, num_of_classes)
        self.softmax1 = nn.Softmax(dim=-1)
        self.linear2 = nn.Linear(512, num_of_classes)
        self.softmax2 = nn.Softmax(dim=-1)
        self.linear3 = nn.Linear(512, num_of_classes)
        self.softmax3 = nn.Softmax(dim=-1)
        self.linear4 = nn.Linear(512, num_of_classes)
        self.softmax4 = nn.Softmax(dim=-1)

        if self.PARAS.DEEPCAPTCHA == 'LSTM':
            # 64*8*24-> 128*4*12
            self.conv4 = myConv2d(64, 128, kernel_size=5, padding='same')
            self.bn4 = nn.BatchNorm2d(128)
            self.act4 = nn.ReLU()
            self.pool4 = nn.MaxPool2d(kernel_size=2)

            # 128*4*12-> 256*2*6
            self.conv5 = myConv2d(128, 256, kernel_size=5, padding='same')
            self.bn5 = nn.BatchNorm2d(256)
            self.act5 = nn.ReLU()
            self.pool5 = nn.MaxPool2d(kernel_size=2)

            self.dropout_lstm = nn.Dropout(0.3)
            self.lstm = nn.LSTM(256*2*6//4, 512, batch_first=True, num_layers=1)

        elif self.PARAS.DEEPCAPTCHA == 'CNN':
            # 64*8*24 -> 64*8*24
            self.dropout_dense = nn.Dropout(0.3)
            self.flatten = nn.Flatten()  # 64*8*24 -> 12288
            self.dense = nn.Linear(12288, 512)
            self.act_dense = nn.ReLU()

        if CBAM:
            self.convBlockAttention = nn.ModuleDict({})
            for layer in PARAS.CBAM:
                self.convBlockAttention[layer] = (ConvBlockAttention(self.CHW[layer]['c'], self.ratio, pool_type)).to(device)

        if EQU:
            self.equalizer = Equalizer(self.CHW['0']['c'], self.CHW['1']['c'], 
                                       self.CHW['0']['h'], self.CHW['1']['h'], connection=PARAS.EQU)
        
        if RES:
            self.convtrans = nn.ModuleDict({})
            for layer_pair in PARAS.RES.split('_'):
                from_layer = self.CHW[layer_pair[0]]
                to_layer = self.CHW[layer_pair[1]]
                # self.convtrans.setdefault(from_layer, {})
                self.convtrans[layer_pair]= ConvTrans(from_layer['c'], to_layer['c'], stride=int(from_layer['h']/to_layer['h'])).to(device)

        if TPS:
            self.thinplatespline = ThinPlateSpline()
        
        if STN:
            self.spatialTransfomer = SpatialTransfomer()

        if AUG:
            aug_paras = self.PARAS.AUG.split('_')
            if len(aug_paras) == 1:
                self.augmentation = augmentations[aug_paras[0]]() 
            else:
                all_paras = aug_paras[1].split('-')
                all_paras = list(map(try_convert2numeric, all_paras))
                self.augmentation = augmentations[aug_paras[0]](*all_paras) 
        
        if FILTER:
            # if int(self.PARAS.FILTER) == 4:
            self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
            self.sigmoid = nn.Sigmoid()
            self.filter1 = nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=5, padding='same'),
                nn.BatchNorm2d(4),
                nn.ReLU(),
            )
            self.filter2 = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=5, padding='same'),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 4, kernel_size=5, padding='same'),
                nn.BatchNorm2d(4),
                nn.ReLU(),
            )
            self.filter3 = nn.Sequential(
                nn.Conv2d(4, 1, kernel_size=5, padding='same'),
                nn.BatchNorm2d(1),
                nn.ReLU()
            )


            # self.filter = nn.Sequential(
            #     nn.Conv2d(1, 4, kernel_size=5, padding='same'),
            #     nn.BatchNorm2d(4),
            #     nn.ReLU(),
            #     nn.Conv2d(4, 16, kernel_size=5, padding='same'),
            #     nn.BatchNorm2d(16),
            #     nn.ReLU(),
            #     nn.Conv2d(16, 4, kernel_size=5, padding='same'),
            #     nn.BatchNorm2d(4),
            #     nn.ReLU(),
            #     nn.Conv2d(4, 1, kernel_size=5, padding='same'),
            #     nn.BatchNorm2d(1),
            #     nn.ReLU()
            # )
            # self.filter = nn.Sequential(
            #     nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            #     nn.BatchNorm2d(16),
            #     nn.ReLU(),
            #     # nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
                # nn.BatchNorm2d(32),
                # nn.ReLU(),
                # nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                # nn.BatchNorm2d(16),
                # nn.ReLU(),
            #     nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            #     nn.BatchNorm2d(1),
            #     nn.ReLU()
            # )
            # elif int(self.PARAS.FILTER) == 3:
            #         self.filter = nn.Sequential(
            #         nn.Conv2d(1, 4, kernel_size=5, padding='same'),
            #         nn.BatchNorm2d(4),
            #         nn.ReLU(),
            #         nn.Conv2d(4, 4, kernel_size=5, padding='same'),
            #         nn.BatchNorm2d(4),
            #         nn.ReLU(),
            #         nn.Conv2d(4, 1, kernel_size=5, padding='same'),
            #         nn.BatchNorm2d(1),
            #         nn.ReLU()
            #     )
            
            # elif int(self.PARAS.FILTER) == 2:
            # self.filter = nn.Sequential(
            #     nn.Conv2d(1, 4, kernel_size=5, padding='same'),
            #     nn.BatchNorm2d(4),
            #     nn.ReLU(),
            #     nn.Conv2d(4, 1, kernel_size=5, padding='same'),
            #     nn.BatchNorm2d(1),
            #     nn.ReLU()
            # )
            # elif int(self.PARAS.FILTER) == 1:
            #         self.filter = nn.Sequential(
            #         nn.Conv2d(1, 1, kernel_size=5, padding='same'),
            #         nn.BatchNorm2d(1),
            #         nn.ReLU()
            #     )
        
        if self.WEIGHTS:
            self.linear_weights = nn.Linear(512, int(PARAS.WEIGHTS.split('-')[1]))
            self.act_weights = nn.Softmax(dim=-1)
        
    def feature_output(self, x, train=False):
        assert(x.dim() == 3 or x.dim() == 4)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if train:
            self.train()
        else:
            self.eval()
        
        if self.AUG:
            x = self.augmentation(x)
            return x

        if self.FILTER:
            x = self.filter(x)
        if self.STN:
            x = self.spatialTransfomer(x)
        return x
        
    def forward(self, x, state='train'):

        if self.AUG and state == 'train':
            # print(state, self.PARAS.AUG)
            x = self.augmentation(x)

        resid = {}
        resid['T'] = x            

        if self.FILTER:
            # x = add_gaussian_noise(x, std_max=0.05)
            # x = add_random_curves(x)
            
            # x = self.filter(x)
            alpha_norm = self.sigmoid(self.alpha)
            x1 = self.filter1(x)
            x2 = self.filter2(x1)
            x = self.filter3(alpha_norm * x1 + (1-alpha_norm) * x2)

        if self.STN:
            # print("ST==T")
            x = self.spatialTransfomer(x)

        
        if self.TPS:
            x, theta = self.thinplatespline(x)
        
        resid['0'] = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        if self.CBAM and '1' in self.PARAS.CBAM:
            x = self.convBlockAttention['1'](x)
        
        if self.EQU:
            x += self.equalizer(resid['0'])

        if self.RES:
            for layer_pair in self.PARAS.RES.split('_'):
                if layer_pair[1] == '1':
                    x += self.convtrans[layer_pair](resid[layer_pair[0]])
        
        resid['1'] = x
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        if self.CBAM and '2' in self.PARAS.CBAM:
            x = self.convBlockAttention['2'](x)
        if self.RES:
            for layer_pair in self.PARAS.RES.split('_'):
                if layer_pair[1] == '2':
                    x += self.convtrans[layer_pair](resid[layer_pair[0]])

        # if self.STN  and self.PARAS.STN == '2':
        #     # print("ST==2")
        #     x = self.spatialTransfomer(x)

        resid['2'] = x
        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)
        if self.CBAM and '3' in self.PARAS.CBAM:
            x = self.convBlockAttention['3'](x)
        if self.RES:
            for layer_pair in self.PARAS.RES.split('_'):
                if layer_pair[1] == '3':
                    x += self.convtrans[layer_pair](resid[layer_pair[0]])
        
        # if self.STN  and self.PARAS.STN == '3':
        #     # print("ST==2")
        #     x = self.spatialTransfomer(x)
        
        if self.PARAS.DEEPCAPTCHA == 'LSTM':
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.act4(x)
            x = self.pool4(x)
            x = self.conv5(x)
            x = self.bn5(x)
            x = self.act5(x)
            x = self.pool5(x)
            x = self.dropout_lstm(x)
            x = self.lstm(x.reshape(x.shape[0], 4, -1))[0]
            x = self.dropout_linear(x)
            out1, out2, out3, out4 = x[:,0,:], x[:,1,:], x[:,2,:], x[:,3,:]

            if self.WEIGHTS:
                weights = self.act_weights(self.linear_weights(out1+out2+out3+out4))

        elif self.PARAS.DEEPCAPTCHA == 'CNN':
            x = self.dropout_dense(x)
            x = self.flatten(x)
            x = self.dense(x)
            x = self.act_dense(x)
            x = self.dropout_linear(x)
            out1 = out2 = out3 = out4 = x

            if self.WEIGHTS:
                weights = self.act_weights(self.linear_weights(x))

        out1 = self.linear1(out1)
        out2 = self.linear2(out2)
        out3 = self.linear3(out3)
        out4 = self.linear4(out4)
        # out1 = self.softmax1(out1)
        # out2 = self.softmax2(out2)
        # out3 = self.softmax3(out3)
        # out4 = self.softmax4(out4)

        if self.WEIGHTS:
            return out1, out2, out3, out4, weights
        elif self.FILTER:
            return out1, out2, out3, out4, alpha_norm
        else:
            return out1, out2, out3, out4, None

Models['DEEPCAPTCHA'] = DeepCAPTCHA

# -------------------------------------------------------------------------------------------------------------------------------------------
def squash(s, dim=-1):
    """
    A non-linear squashing function to ensure that vector length is between 0 and 1. (Eq. (1) from the original paper)
    :param s: A weighted sum over all prediction vectors.
    :param dim: Dimension along which the length of a vector is calculated.
    :return: Squashed vector.
    """
    squared_length = torch.sum(s ** 2, dim=dim, keepdim=True)

    return squared_length / (1 + squared_length) * s / (torch.sqrt(squared_length) + 1e-8)  # avoid zero denominator


class PrimaryCapsules(nn.Module):
    def __init__(self, input_shape, out_channels, dim_caps=8, kernel_size=9, stride=2, padding=0, coord=False, attention=False):
        """
        Initialise a Primary Capsules layer.
        :param input_shape: The shape of the inputs going in to the network.
        :param out_channels: The number of output channels.
        :param dim_caps: The dimension of a capsule vector.
        :param kernel_size: The size of convolutional kernels
        :param stride: The amount of movement between kernel applications
        :param padding: The addition of pixels to the edge of the feature maps
        """
        super(PrimaryCapsules, self).__init__()
        self.input_shape = input_shape
        self.out_channels = out_channels
        self.dim_caps = dim_caps
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.coord = coord
        self.attention = attention
        # initialize a module dict, which is effectively a dictionary that can collect layers
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        """
        Build neural networks and print out the shape of feature maps.
        """
        with torch.no_grad():
            # print("Building a Primary Capsules layer with input shape: ", self.input_shape)
            x = torch.zeros(self.input_shape)
            # Compute the number of capsules in the Primary Capsules layer
            self.num_caps = int(self.out_channels / self.dim_caps)
            self.layer_dict['conv'] = nn.Conv2d(in_channels=self.input_shape[1],
                                                out_channels=self.out_channels,
                                                kernel_size=self.kernel_size,
                                                stride=self.stride,
                                                padding=self.padding)

            out = self.layer_dict['conv'](x)
            # print("The shape of feature maps after convolution: ", out.shape)
            # Group dim_caps pixel values together across channels as elements of a capsule vector
            h = out.size(2)
            w = out.size(3)
            out = out.view(out.size(0), self.num_caps, self.dim_caps, out.size(2), out.size(3))
            out = out.permute(0, 1, 3, 4, 2).contiguous()
            # attention for capsule type
            if self.attention:
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
                mask = self.avg_pool(out).squeeze()
                self.fc_1 = nn.Linear(mask.size(-1), 16)
                mask = F.relu(self.fc_1(mask))
                self.fc_2 = nn.Linear(16, out.size(1))
                mask = torch.sigmoid(self.fc_2(mask))
                mask = mask.view(out.size(0), out.size(1), 1, 1, 1)
                out = mask * out

            # if self.coord:
            #     out = coord_addition_cap(out)
            # Sequentialise capsule vectors
            out = out.view(out.size(0), -1, out.size(-1))
            out = squash(out)
            # print("The shape of the layer output: ", out.shape)

    def forward(self, x):
        out = self.layer_dict['conv'](x) # [4, 128, 8, 24] - [4, 256, 4, 12]
        h = out.size(2)
        w = out.size(3)
        out = out.view(out.size(0), self.num_caps, self.dim_caps, out.size(2), out.size(3))
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        _, _, h, w, dim = out.shape

        # attention for capsule type
        if self.attention:
            mask = out.sum((4, 3, 2)) / (h * w * dim)
            mask = F.relu(self.fc_1(mask))
            mask = torch.sigmoid(self.fc_2(mask))
            mask = mask.view(out.size(0), out.size(1), 1, 1, 1)
            out = mask * out + 1e-6

        # if self.coord:
        #     out = coord_addition_cap(out)
        out = out.view(out.size(0), -1, out.size(-1))

        return squash(out)


class RoutingCapsules(nn.Module):
    def __init__(self, input_shape, num_caps, dim_caps, num_iter, ice: torch.device):
        """
        Initialise a Routing Capsule layer. We only have one such layer (DigitCaps) in the original paper.
        :param input_shape: The input shape of the Routing Capsules layer. (batch_size, num_caps_input, dim_caps_input)
        :param num_caps: The number of output capsules.
        :param dim_caps: The dimension of each output capsule.
        :param num_iter: The number of routing iterations.
        :param device: Store tensor variables in CPU or GPU.
        """
        super(RoutingCapsules, self).__init__()
        self.input_shape = input_shape
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.num_iter = num_iter
        self.device = device
        self.layer_dict = nn.ModuleDict()
        self.build_module()

    def build_module(self):
        with torch.no_grad():
            # print("Building a RoutingCapsules layer with input shape: ", self.input_shape)

            x = torch.zeros(self.input_shape)  # (batch_size, num_caps_input, dim_caps_input)
            x = x.unsqueeze(1).unsqueeze(4)    # (batch_size, 1, num_caps_input, dim_caps_input, 1)
            # Weight matrix (1, num_caps_output, num_caps_input, dim_caps_output, dim_caps_input)
            self.W = nn.Parameter(0.01 * torch.randn(1, self.num_caps, self.input_shape[1],
                                                     self.dim_caps, self.input_shape[2]))
            # Prediction vectors (batch_size, num_caps_output, num_caps_input, dim_caps_output, 1)
            u_hat = torch.matmul(self.W, x)
            u_hat = u_hat.squeeze(-1)
            # print("The shape of prediction vectors: ", u_hat.shape)

            # One routing iteration
            # print("Test one routing iteration")
            # Initial logits (batch_size, num_caps_output, num_caps_input, 1)
            b = torch.zeros(*u_hat.size()[:3], 1)
            # Coupling coefficient
            c = F.softmax(b, dim=1)
            # Weighted sum (batch_size, num_caps_output, dim_caps_output)
            s = (c * u_hat).sum(dim=2)
            # Capsule vector output
            v = squash(s)
            # print("The shape of capsule ouput: ", v.shape)
            # Update b
            b += torch.matmul(u_hat, v.unsqueeze(-1))
            # print("Routing iteration completed")

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(4)
        u_hat = torch.matmul(self.W, x)
        u_hat = u_hat.squeeze(-1)
        # We do not want to create computational graph during the iteration
        # with torch.no_grad():
        b = torch.zeros(*u_hat.size()[:3], 1).to(x.device)
        # Routing iterations except for the last one
        for r in range(self.num_iter - 1):
            c = F.softmax(b, dim=1)
            s = (c * u_hat).sum(dim=2)
            v = squash(s)
            b = b + torch.matmul(u_hat, v.unsqueeze(-1))

        # Connect the last iteration to computational graph
        c = F.softmax(b, dim=1)
        s = (c * u_hat).sum(dim=2)
        # v = squash(s)
        v = s

        return v

class DynamicCapsules(nn.Module):
    def __init__(self, PARAS, input_shape=(1, INPUT_CHANNEL, 64, 192), num_channels_primary=256, dim_caps_primary=8,
                 num_classes=26, dim_caps_output=64, num_iter=3, device=device, kernel_size=4, 
                 dropout=True, coord=False, attention=False, stride=1, **model_config):
        """
        Capsule networks with dynamic routing

        :param input_shape: Input image shape (batch_size, # channels, h, w)
        :param num_channels_primary: The number of channels of feature maps going in and out of Primary Capsules layer
        :param dim_caps_primary: The dimension of each capsule in the Primary Capsule layer
        :param num_classes: The number of classes of the dataset
        :param dim_caps_output: The dimension of each capsule in the Digit Capsules layer
        :param num_iter: The number of routing iteration in the Digit Capsule layer
        :param device: CPU or GPU
        :param kernel_size: The size of the kernel in Conv1 and Primary Capsule layer
        """
        super(DynamicCapsules, self).__init__()
        self.AUG = model_config.get('AUG', None)
        self.input_shape = input_shape
        self.num_channels_primary = num_channels_primary
        self.dim_caps_primary = dim_caps_primary
        self.num_classes = num_classes
        self.dim_caps_output = dim_caps_output
        self.num_iter = num_iter
        self.device = device
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = dropout
        self.in_channel = INPUT_CHANNEL
        if dropout:
            self.Dropout = nn.Dropout(p=0.3)
        self.coord = coord
        self.attention = attention
        self.relu = nn.ReLU(inplace=True)
        self.layer_dict = nn.ModuleDict()
        self.build_module()

    def build_module(self):
        with torch.no_grad():
            # print("Building Dynamic Capsule Networks with input shape: ", self.input_shape)
            x = torch.zeros(self.input_shape)
            # 1,64,192 -> 32,32,96
            self.layer_dict['conv1'] = nn.Conv2d(self.input_shape[1], self.num_channels_primary//8, 4, stride=2, padding=1)
            self.bn_conv1 = nn.BatchNorm2d(self.num_channels_primary//8)
            x = self.layer_dict['conv1'](x)
            x = self.relu(self.bn_conv1(x))
            # 32,32,96 -> 64,16,48
            self.layer_dict['conv2'] = nn.Conv2d(self.num_channels_primary//8, self.num_channels_primary//4, 4, stride=2, padding=1)
            self.bn_conv2 = nn.BatchNorm2d(self.num_channels_primary//4)
            out = self.layer_dict['conv2'](x)
            out = self.relu(self.bn_conv2(out))
            # 64,16,32 -> 128,8,24
            self.layer_dict['conv3'] = nn.Conv2d(self.num_channels_primary//4, self.num_channels_primary//2, 4, stride=2, padding=1)
            self.bn_conv3 = nn.BatchNorm2d(self.num_channels_primary//2)
            out = self.layer_dict['conv3'](out)
            out = self.relu(self.bn_conv3(out))

            self.layer_dict['primarycaps'] = PrimaryCapsules(out.shape, self.num_channels_primary,
                                                             self.dim_caps_primary, self.kernel_size,
                                                             coord=self.coord, attention=self.attention, padding=1)
            out = self.layer_dict['primarycaps'](out)
            # Only output 4 capsules because there are only 4 digits in the image
            self.layer_dict['digitcaps'] = RoutingCapsules(out.shape, 4, self.dim_caps_output,
                                                           self.num_iter, self.device)
            out = self.layer_dict['digitcaps'](out)  # (batch_size, num_caps_output = 4, dim_caps_output = 16)
            # print("The shape of Digit Capsules layer output is: ", out.shape)
            self.layer_dict['fcc'] = nn.Linear(self.dim_caps_output, self.num_classes)
            out = self.layer_dict['fcc'](out) # (batch_size, num_caps_output = 4, num_classes = 10)


    def forward(self, x, state='train'):
        if self.AUG and (state == 'train' or self.PARAS.AUG in ['mixchannel']):
            # print(state, self.PARAS.AUG)
            x = self.augmentation(x)

        out = self.layer_dict['conv1'](x) # 1,64,192 - 32,32,96
        out = self.relu(self.bn_conv1(out))
        out = self.layer_dict['conv2'](out) # 32,32,96 -> 64,16,48
        out = self.relu(self.bn_conv2(out))
        out = self.layer_dict['conv3'](out) # 64,16,48 -> 128,8,24
        out = self.relu(self.bn_conv3(out))
        out = self.layer_dict['primarycaps'](out)

        if self.dropout:
            dropout_mask = torch.ones(out.size()[:2]).to(out.device)
            dropout_mask = self.Dropout(dropout_mask)
            dropout_mask = dropout_mask.unsqueeze(-1)
            out = out * dropout_mask

        out = self.layer_dict['digitcaps'](out) # (batch_size, num_caps_output = 10, dim_caps_output = 16)
        if self.dropout:
            out = self.Dropout(out)
        out = self.layer_dict['fcc'](out)
        out1 = out[:,0,:]
        out2 = out[:,1,:]
        out3 = out[:,2,:]
        out4 = out[:,3,:]
        # return out, reconstructions
        return out1, out2, out3, out4, None

Models['CAPSULE'] = DynamicCapsules

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

class MultiKernelConvBlock(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(MultiKernelConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//6, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels//6)
        self.conv3 = nn.Conv2d(in_channels, out_channels//6*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels//6*4)
        self.conv5_decomposed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//6, kernel_size=3, padding=1),
            nn.Conv2d(out_channels//6, out_channels//6, kernel_size=3, padding=1)
        )
        self.bn5_decomposed = nn.BatchNorm2d(out_channels//6)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)

        
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        
        x5 = self.conv5_decomposed(x)
        x5 = self.bn5_decomposed(x5)
        x5 = F.relu(x5)
        
        out = torch.cat((x1, x3, x5), 1)

        x = self.conv(out)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ConvBlockx2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockx2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class PredictiveConvBlock(nn.Module):
    def __init__(self, in_channels, out_features, num_groups=4):
        super(PredictiveConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(in_channels)
        self.group_conv = nn.Conv2d(48, 48, kernel_size=1, padding=0, groups=num_groups)
        self.group_bn = nn.BatchNorm2d(48)
        self.dropout = nn.Dropout(0.3)
        self.conv1 = nn.Conv2d(48*4*512, out_features, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(48*4*512, out_features, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(48*4*512, out_features, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(48*4*512, out_features, kernel_size=1, padding=0)

        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        self.softmax3 = nn.Softmax(dim=-1)
        self.softmax4 = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x) # (512,4,48)
        
        # 转置通道和宽度维度
        x = x.permute(0, 3, 2, 1) # (512,4,48) -> (48,4,512)

        # 分组卷积
        x = self.group_conv(x) # (48,4,512) -> (48,4,512)
        x = self.group_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        # 再次转置维度
        x1 = self.conv1(x.reshape(-1, 48*4*512, 1, 1))
        x2 = self.conv2(x.reshape(-1, 48*4*512, 1, 1))
        x3 = self.conv3(x.reshape(-1, 48*4*512, 1, 1))
        x4 = self.conv4(x.reshape(-1, 48*4*512, 1, 1))
        # CE
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)
        # BCE
        # x1 = self.softmax1(x1.view(x1.size(0), -1))
        # x2 = self.softmax2(x2.view(x2.size(0), -1))
        # x3 = self.softmax3(x3.view(x3.size(0), -1))
        # x4 = self.softmax4(x4.view(x4.size(0), -1))

        return x1, x2, x3, x4, None


class ConvNet(nn.Module):
    def __init__(self, PARAS=None, **model_config):
        super(ConvNet, self).__init__()
        self.AUG = model_config.get('AUG', None)
        self.in_channel = INPUT_CHANNEL
        self.multi_kernel_conv_block1 = MultiKernelConvBlock(INPUT_CHANNEL, 48)  
        self.multi_kernel_conv_block2 = MultiKernelConvBlock(48, 96)  # 假设输入图像为3通道
        self.blockx21 = ConvBlockx2(96, 128)
        self.blockx22 = ConvBlockx2(128, 256)
        self.blockx23 = ConvBlockx2(256, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.groups = PARAS.GROUPS if PARAS and PARAS.GROUP else 4
        self.predictive_conv_block = PredictiveConvBlock(512, 26, num_groups=self.groups)  # 假设有26个输出类别和5个分组
        

    def forward(self, x, state='train'):
        if self.AUG and (state == 'train' or self.PARAS.AUG in ['mixchannel']):
            # print(state, self.PARAS.AUG)
            x = self.augmentation(x)

        x = self.multi_kernel_conv_block1(x) # (1,64,192) -> (48,64,192)
        x = F.max_pool2d(x, kernel_size=2, stride=2) # (48,64,192) -> (48,32,96)
        x = self.multi_kernel_conv_block2(x) # (48,32,96) -> (96,32,96)
        x = F.max_pool2d(x, kernel_size=2, stride=2) # (96,32,96) -> (96,16,48)
        x = self.blockx21(x) # (96,16,48) -> (128,16,48) 
        x = F.max_pool2d(x, kernel_size=(2,1), stride=(2,1)) # (128,16,48) -> (128,8,48)
        x = self.blockx22(x)  # (128,8,48) -> (256,8,48)
        x = F.max_pool2d(x, kernel_size=(2,1), stride=(2,1)) # (256,8,48) -> (256,4,48)
        x = self.blockx23(x) # (256,4,48) -> (512,4,48)
        x = self.dropout1(x)
        x = self.predictive_conv_block(x)
        return x

Models['CONVNET'] = ConvNet

# ---------------------------------------------------------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, num_of_classes=26) -> None:
        super(Model, self).__init__()
        self.num_of_classes = num_of_classes
        # 1,64,192
        self.conv1 = nn.Conv2d(INPUT_CHANNEL, 32, kernel_size=3, padding='same')
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2) 
        # 32*32*96 
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, padding='same')
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 48*16*48 
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, padding='same')
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # 64*8*24
        self.conv4 = nn.Conv2d(64, 96, kernel_size=3, padding='same')
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        # 96*4*12
        self.conv5 = nn.Conv2d(96, 128, kernel_size=3, padding='same')
        self.act5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=(2,3))
        # 128*2*4
        
        self.rnn1 = nn.LSTM(128*2, 128, bidirectional=True, batch_first=True)
        # self.rnn2 = nn.LSTM(128*2, 128, bidirectional=True, batch_first=True)
        self.dense = nn.Linear(2 * 128, 26)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = self.pool4(self.act4(self.conv4(x)))
        x = self.pool5(self.act5(self.conv5(x)))
        # b,128,2,4
        x = x.view(x.size(0), x.size(1) * x.size(2), -1)
        # b,128*2,4
        x = x.permute(0,2,1)
        # b,4,128*2

        x, _ = self.rnn1(x.reshape(x.shape[0], 4, -1))
        x = self.dense(x)
        return x

class MultiView(nn.Module):
    def __init__(self, PARAS=None, **model_config):
         super(MultiView, self).__init__()
         self.in_channel = INPUT_CHANNEL
         self.model1 = Model()
         self.model2 = Model()
    
    def forward(self, img2):
        img1 = img2[:,0]
        img2 = img2[:,1]
        x = self.model1(img1)
        y = self.model2(img2)
        out = (x+y)/2
        return out[:,0], out[:,1], out[:,2], out[:,3] , None

Models['MULTIVIEW'] = MultiView

# --------------------------------------------------------------------------------------------------------------

class VeriBypasser(nn.Module):
    def __init__(self, PARAS=None, **model_config):
        super(VeriBypasser, self).__init__()
        self.in_channel = INPUT_CHANNEL
        self.AUG = model_config.get('AUG', None)
        self.layer1 = nn.Sequential(
            nn.Conv2d(INPUT_CHANNEL, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear((192//8)*(64//8)*64, 1024),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.rfc = nn.Sequential(
            nn.Linear(1024, 26*4),
        )

    def forward(self, x, state='train'):
        if self.AUG and (state == 'train' or self.PARAS.AUG in ['mixchannel']):
            # print(state, self.PARAS.AUG)
            x = self.augmentation(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        out1, out2, out3, out4 = out[:,:26], out[:,26:52], out[:,52:78], out[:,78:104]
        return out1, out2, out3, out4, None

Models['VERIBYPASSER'] = VeriBypasser
# --------------------------------------------------------------------------------------------------------------


def load_model_nopara(pthfilename):
    pthfilename = os.path.basename(pthfilename)
    Pstring = get_Pstring(pthfilename)
    Pkey = get_Pkey(pthfilename)
    P = str_to_paras(Pstring)
    Model = Models[pthfilename.split("@")[0].split('_')[0].upper()]
    model = Model(PARAS=P, **Pkey)
    return model


def measure_inference_time(model, input_image, num_iterations=1000, mode='cpu'):

    assert mode == 'cpu' or mode == 'gpu'

    if mode == 'gpu':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    # 预热模型
    with torch.no_grad():
        for _ in range(10):
            model(input_image)

    # 记录开始时间
    if mode == 'gpu':
        start_event.record()
    else:
        start_time = time.time()

    # 进行多次推理以获得平均时间
    with torch.no_grad():
        for _ in range(num_iterations):
            model(input_image)

    # 记录结束时间
    if mode == 'gpu':
        end_event.record()
        end_event.synchronize()
    else:
        end_time = time.time()

    # 计算总时间
    total_time_ms = start_event.elapsed_time(end_event) if mode == 'gpu' else (end_time - start_time)*1000 

    # 计算平均时间
    average_time_ms = total_time_ms / num_iterations  # 转换为毫秒
    return average_time_ms



def compute_parameters(filepath):
    if os.path.exists("/content"):
        filepath = "/content/drive/MyDrive/models"
        data_path = os.path.join("/content", "mcaptcha2")

    paths = glob.glob(os.path.join(filepath, "*.pth"))
    for path in paths:
        model = load_model_nopara(path)
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state'])
        total_params = sum(p.numel() for p in model.parameters())
        print(os.path.basename(path), "Parameters:", total_params)


def computation_FPS(filepath, data_path, recursive, inclusions, inlogic, exclusions, exlogic, **kwargs):
    if os.path.exists("/content"):
        filepath = "/content/drive/MyDrive/models"
        data_path = os.path.join("/content", "mcaptcha2")

    dl_gray, _ = data_generate(batch_size=64, split=0.8, data_path=data_path, colormode='gray')
    dl_RGB, _ = data_generate(batch_size=64, split=0.8, data_path=data_path, colormode='rgb')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    paths = filter_files(filepath, "*.pth", recursive, inclusions, inlogic, exclusions, exlogic)
    for path in paths:
        model = load_model_nopara(path)
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state'])
        model.to(device)
        dl = dl_gray if model.in_channel == 1 else dl_RGB

        # FPS
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()

        with torch.no_grad():
            for x_data, y_data in dl:
                _ = model(x_data.to(device))

        torch.cuda.synchronize() if device.type == "cuda" else None   
        end_time = time.time() 
        inference_time = end_time - start_time
        total_imgs = len(dl.dataset)
        fps = total_imgs / inference_time
        print(os.path.basename(path), "FPS:", fps)


def computation_GFLOPs_Parameters(filepath, data_path, only, recursive, inclusions, inlogic, exclusions, exlogic, timeperimage=False):
    if os.path.exists("/content"):
        filepath = "/content/drive/MyDrive/models"
        data_path = os.path.join("/content", "mcaptcha2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    paths = filter_files(filepath, "*.pth", recursive, inclusions, inlogic, exclusions, exlogic)
    for path in paths:
        model = load_model_nopara(path)
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state'])
        model.to(device)

        # GFLOPS
        input = torch.randn(1, 3, 64, 192).to(device) if model.in_channel ==3 else torch.randn(1, 1, 64, 192).to(device)
        # flops = FlopCountAnalysis(model, input)
        # gflops = flops.total() / 1e9
        if only.upper() in ["AASR", "A"]: 
            print(os.path.basename(path) + '\t' + str(round(sum(checkpoint['test_accuracy']['average'][-10:])/10, 3)))
        elif only.upper() in ["PARAMETERS", "P"]:
            total_params = sum(p.numel() for p in model.parameters())
            print(os.path.basename(path) + '\t' + str(total_params))
        else:
            flops, params = profile(model, inputs=(input,))
            flops, params = clever_format([flops, params])
            print(os.path.basename(path), "FLOPs:", flops, "Params:", params, "AASR:", round(sum(checkpoint['test_accuracy']['average'][-8:])/8, 3), 
            'Convergence Epoch:', round(int(len(checkpoint['test_accuracy']['average']) - (torch.tensor(checkpoint['test_accuracy']['average']) > floor(sum(100*checkpoint['test_accuracy']['average'][-10:])/10)/100).sum())/10)*10
            )
        if timeperimage:
            tperimage =measure_inference_time(model, input)
            print("time per image:", tperimage)

# AARI_bf19f7
# ABBW_ffbae976-269f-4a5e-9d4f-dcb67bcc9fbc
# AATL_897f4fef-bfe6-4b31-8dde-41791ee2cc0e
# AARI_bf46e264-5234-4b56-80e7-09ced3c319f7
# ABCH_615b3fe2-2fdb-451e-aa72-7a2f84b17ca4
def show_grad_cam(filepath, recursive, inclusions, inlogic, exclusions, exlogic, target_category=0,
                  img_path=r"D:\my-code\data\dataset\2\dataset\mcaptcha_denoise\AARI_bf19f7.jpg"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    paths = filter_files(filepath, "*.pth", recursive, inclusions, inlogic, exclusions, exlogic)
    for path in paths:
        model = load_model_nopara(path)
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state'])
        model.to(device)

        target_layers = [model.act5]
        if model.in_channel == 3:
            data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.456], [0.224])])

        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img_RGB = Image.open(img_path).convert('RGB')
        img_RGB = np.array(img_RGB, dtype=np.uint8)
        if model.in_channel == 1:
            img = Image.open(img_path).convert('L')
            img = np.array(img, dtype=np.uint8)
        else:
            img = img_RGB
        # [C, H, W]
        img_tensor = data_transform(img)
        # expand batch dimension
        # [C, H, W] -> [N, C, H, W]
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        input_tensor = torch.unsqueeze(img_tensor, dim=0)
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img_RGB.astype(dtype=np.float32) / 255.,
                                        grayscale_cam,
                                        use_rgb=True)
        plt.imshow(visualization)
        save_path = os.path.basename(img_path)
        plt.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()




if __name__ == "__main__":
    from thop import profile, clever_format
    # model = ConvNet(26)

    # # 假设输入一个64x160的彩色图像
    # input_tensor = torch.randn(1, 3, 64, 192)

    # # 前向传播
    # output = model(input_tensor)
    # print(output.shape)  # 应输出预测序列的维度，例如：(1, 25, 160)

    # computation_FPS(filepath=r"D:\Results\FEGAN result\preprocessing_compare\mcaptcha")
    computation_GFLOPs_Parameters(
    # computation_FPS(
        filepath=r"D:\Results\FEGAN result\preprocessing_compare\mcaptcha\新建文件夹", only='',
        inclusions=['gb'], inlogic='and', exclusions=[], exlogic='or',
        recursive=False, data_path=r'D:\my-code\data\sample', timeperimage=True
    )



    # AARI_bf19f7
    # ABBW_ffbae976-269f-4a5e-9d4f-dcb67bcc9fbc
    # AATL_897f4fef-bfe6-4b31-8dde-41791ee2cc0e
    # AARI_bf46e264-5234-4b56-80e7-09ced3c319f7
    # ABCH_615b3fe2-2fdb-451e-aa72-7a2f84b17ca4
    # show_grad_cam(
    #     filepath=r"D:\Results\FEGAN result\compare_mcaptcha\results", 
    #     # inclusions=['FE','LSTM'], inlogic='and', exclusions=[], exlogic='or', recursive=False, 
    #     inclusions=['LSTM'], inlogic='and', exclusions=['FE'], exlogic='or', recursive=False, 
    #     target_category=0, 
    #     img_path=r"D:\my-code\data\dataset\2\dataset\test\aug\ABWG_1410cf03-b8d5-4b30-8f13-579af5defbea.jpg")



    

    
