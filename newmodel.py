import torch
import torch.nn as nn
import cv2  
import numpy as np  
import matplotlib.pyplot as plt
from dataset import *


def filter_graycale(filtertype='median'):
    if filtertype == 'median':
        pass



class Model(nn.Module):
    def __init__(self, num_of_classes=26) -> None:
        super(Model, self).__init__()
        self.num_of_classes = num_of_classes
        # 1,64,192
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
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

        # x = x.permute(2, 0, 1)
        # 4,b,128*2

        # # 6,b,128*2
        # x, _ = self.rnn1(x.reshape(x.shape[0], 4, -1))
        # x, _ = self.rnn2(x)

        x, _ = self.rnn1(x.reshape(x.shape[0], 4, -1))
        x = self.dense(x)
        return x

class Multiview(nn.Module):
    def __init__(self):
         super(Multiview, self).__init__()
         self.model1 = Model()
         self.model2 = Model()
    
    def forward(self, img2):
        img1 = img2[:,0]
        img2 = img2[:,1]

        x = self.model1(img1)
        y = self.model2(img2)

        out = (x+y)/2
        return out

train_loader, test_loader = data_generate(batch_size=7)
img2s, labels = next(iter(test_loader))

# x = torch.randn(7,1,64,192)
model = Multiview()
y = model(img2s)
print(y.shape)



# -*- coding: utf-8 -*-
# from dataset import *
# criterion = nn.CTCLoss(reduction='sum', zero_infinity=True)
# train_loader, test_loader = data_generate(4)

# for imgs, label in train_loader:
#     img2s, labels = next(iter(test_loader))
#     imgs = img2s[0]
#     model_median = Model()
#     # model_lilateral = Model()
#     targets = torch.argmax(labels, dim=-1)
#     logits = model_median(imgs)
#     # y = model_lilateral(imgs[1])
#     log_probs = torch.nn.functional.log_softmax(logits, dim=2)
#     batch_size = imgs.size(0)
#     input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
#     target_lengths = torch.LongTensor([4] * batch_size)
#     loss = criterion(log_probs, targets, input_lengths, target_lengths)
#     print(loss)


