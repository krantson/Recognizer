import torch.nn as nn
import torch.nn.functional as F
import torch

class Equalizer(nn.Module):
    def __init__(self, in_channels, out_channels, in_height, out_height, connection='cascade') -> None:
        super(Equalizer, self).__init__()
        self.stride = int(in_height/out_height)
        self.connection = connection
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=self.stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if self.connection == 'cascade':
            in_channels_layer2 = out_channels
            stride = 1
        elif self.connection == 'cross':
            in_channels_layer2 = in_channels
            stride = self.stride
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_layer2, out_channels=out_channels, kernel_size=9, stride=stride, padding=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if self.connection == 'cascade':
            in_channels_layer3 = out_channels
            stride = 1
        elif self.connection == 'cross':
            in_channels_layer3 = in_channels
            stride = self.stride
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_layer3, out_channels=out_channels, kernel_size=5, stride=stride, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        
    def forward(self, x):
        if self.connection == 'cascade':
            out1 = self.layer1(x)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            return out1 + out2 + out3
        elif self.connection == 'cross':
            out1 = self.layer1(x)
            out2 = self.layer2(x)
            out3 = self.layer3(x)
            return out1 + out2 + out3
