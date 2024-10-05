import torch.nn as nn
import torch.nn.functional as F
import torch
import glob
import os
import cv2
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
from utils import str_to_model_kwargs, str_to_paras


# class SpatialTransfomer(nn.Module):
#     def __init__(self, position='T'):
#         super(SpatialTransfomer, self).__init__()
#         self.position = position
#         if position == 'T' or position == 'TF':
#             # 1*64*192
#             self.localization = nn.Sequential(
#                 nn.Conv2d(1, 8, kernel_size=7), # B*1*64*192 -> B*8*58*186
#                 nn.MaxPool2d(2, stride=2), # B*8*57*185 - B*8*29*93
#                 nn.ReLU(True),
#                 nn.Conv2d(8, 16, kernel_size=5), # B*8*29*93 - B*16*25*89
#                 nn.MaxPool2d(2, stride=2), # B*16*25*89 - B*16*12*44
#                 nn.ReLU(True),
#                 nn.Conv2d(16, 32, kernel_size=5), # B*16*12*44 - B*32*8*40
#                 nn.MaxPool2d(2, stride=2), # B*32*8*40 - B*32*4*20
#                 nn.ReLU(True),
#                 nn.Conv2d(32, 48, kernel_size=3), # B*32*4*20 - B*48*2*18
#                 nn.MaxPool2d(2, stride=2), # B*48*2*38 - B*48*1*9
#                 nn.ReLU(True),
#             )

#             self.fc_loc = nn.Sequential(
#                 nn.Flatten(),
#                 nn.Linear(48*1*9, 32),
#                 nn.ReLU(True),
#                 nn.Linear(32, 3*2)
#             )

#         elif str(position) == '2':
#             # 48*16*48
#             self.localization = nn.Sequential(
#                 nn.Conv2d(48, 56, kernel_size=4, stride=2, padding=1), # B*48*16*48 -> B*56*8*24
#                 nn.MaxPool2d(2, stride=2), # B*56*8*24 - B*56*4*12
#                 nn.ReLU(True),
#                 nn.Conv2d(56, 64, kernel_size=4, stride=2, padding=1), # B*56*4*12 - B*64*2*6
#                 nn.MaxPool2d(2, stride=2), # B*64*2*6 - B*64*1*3
#                 nn.ReLU(True)
#             )

#             self.fc_loc = nn.Sequential(
#                 nn.Flatten(),
#                 nn.Linear(64*1*3, 32),
#                 nn.ReLU(True),
#                 nn.Linear(32, 3*2)
#             )
        
#         elif str(position) == '3':
#             # 64*8*24
#             self.localization = nn.Sequential(
#                 nn.Conv2d(64, 72, kernel_size=4, stride=2, padding=1), # B*64*8*24 -> B*72*4*12
#                 nn.MaxPool2d(2, stride=2), # B*72*4*12 - B*72*2*6
#                 nn.ReLU(True),
#                 nn.Conv2d(72, 80, kernel_size=4, stride=2, padding=1), # B*72*2*6 - B*80*1*3
#                 nn.ReLU(True)
#             )

#             self.fc_loc = nn.Sequential(
#                 nn.Flatten(),
#                 nn.Linear(80*1*3, 32),
#                 nn.ReLU(True),
#                 nn.Linear(32, 3*2)
#             )

#         self.fc_loc[3].weight.data.zero_()
#         self.fc_loc[3].bias.data.copy_(torch.tensor([1,0,0,0,1,0]))
    
#     def stn(self, x, theta=None):
#         if theta == None:
#             xs = self.localization(x)
#             theta = self.fc_loc(xs)
#             theta = theta.view(-1, 2, 3)

#         grid = F.affine_grid(theta, x.size(), align_corners=True)   
#         x = F.grid_sample(x, grid, align_corners=True)
#         return x

#     def forward(self, x):
#         x = self.stn(x)
#         return x




class SpatialTransfomer(nn.Module):
    def __init__(self):
        super(SpatialTransfomer, self).__init__()
        
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=9, padding="same"), # B*1*64*192 -> B*8*64*192
            nn.MaxPool2d(2, stride=2), # B*8*64*192 - B*8*32*96
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=7, padding='same'), # B*8*32*96 - B*16*32*96
            nn.MaxPool2d(2, stride=2), # B*16*32*96 - B*16*16*48
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=5, padding='same'), # B*16*16*48 - B*16*16*48
            nn.MaxPool2d(2, stride=2), # B*16*16*48 - B*16*8*24
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, padding='same'), # B*16*8*24 - B*32*8*24
            nn.MaxPool2d(2, stride=2), # B*32*8*24 - B*32*4*12
            nn.ReLU(True),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(32*4*12, 32),
            nn.ReLU(True),
            nn.Linear(32, 3*2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0]))
    
    def stn(self, x, theta=None):
        if theta == None:
            xs = self.localization(x)
            xs = xs.view(-1, 32*4*12)
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)   
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, x):
        x = self.stn(x)
        return x



def show_stn_output(model, data_path='train', img_format="*.jpg", seed=None, num_figs=2):
    if seed != None:
        random.seed(seed)
    toTensor = transforms.ToTensor()
    img_paths = glob.glob(os.path.join(data_path, img_format))
    random.shuffle(img_paths)

    img_inputs, img_outputs = [], []
    theta = torch.tensor([[[1,0,0], [0.5,1,0]]])
    for img_path in img_paths[:num_figs]:
        img = toTensor(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY))
        img_inputs.append(img)
        img_outputs.append(model.stn(img.unsqueeze(0), theta).squeeze(0))

    imgs = make_grid(img_inputs + img_outputs, nrow=num_figs, padding=20)
    imgs = np.transpose(imgs.numpy(), (1,2,0))

    plt.imshow(imgs)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    model = SpatialTransfomer()
    show_stn_output(model)