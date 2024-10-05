import torch.nn as nn
import torch
from dataset2 import *

class CRNN(nn.Module):

    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = \
            self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)

        conv = self.cnn(images) # (b, 512, 3, 47)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width) # (b, 512*3, 47)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature) # (47, b, 512*3)
        seq = self.map_to_seq(conv) # (47, 3, 64)

        recurrent, _ = self.rnn1(seq) # (47,3,2*256)
        recurrent, _ = self.rnn2(recurrent) # (47,3,2*256)

        output = self.dense(recurrent) # (47, 3, 26)
        return output  # shape: (seq_len, batch, num_class)

train_loader, _ = data_generate(batch_size=3, num_of_letters=4, split=0.5, data_path=r'D:\learning-code\data\good', 
                  charsets='ABCDEFGHIJKLMNOPQRSTUVWXYZ', img_row=64, img_column=192, img_format='*.jpg')
criterion = nn.CTCLoss(reduction='sum', zero_infinity=True)
crnn = CRNN(img_channel=1, img_height=64, img_width=192, num_class=26)
# images = torch.randn(3,1,64,192)
(images, labels) = next(iter(train_loader))
targets = torch.argmax(labels, dim=-1)
logits = crnn(images)
log_probs = torch.nn.functional.log_softmax(logits, dim=2)
batch_size = images.size(0)
input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
target_lengths = torch.LongTensor([4] * batch_size)
loss = criterion(log_probs, targets, input_lengths, target_lengths)





