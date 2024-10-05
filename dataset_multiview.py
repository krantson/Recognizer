from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
import string
import glob
import cv2
import os
import torch
import random
import numpy as np


random.seed(0)
torch.manual_seed(0)

class CaptchaDataset(Dataset):
    def __init__(self, img_paths, labels, transform, split, train=True):
        super().__init__()
        if train:
            self.img_paths = img_paths[:int(len(img_paths) * split)]
            self.labels = labels[:int(len(img_paths) * split)]
        else:
            self.img_paths = img_paths[int(len(img_paths) * split):]
            self.labels = labels[int(len(img_paths) * split):]
        self.transform = transform
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        img = cv2.imread(img_path)  # (h, w, c), BGR, ToTensor 不转换颜色通道
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_median = cv2.medianBlur(img, 5)
        img_bilateral = cv2.bilateralFilter(img, 15, 150, 150)

        # img_median = cv2.cvtColor(img_median, cv2.COLOR_BGR2GRAY) # (h, w, c)->(h, w)
        img_median = self.transform(img_median) # (h, w) -> (1, h, w)

        # img_bilateral = cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2GRAY) # (h, w, c)->(h, w)
        img_bilateral = self.transform(img_bilateral) # (h, w) -> (1, h, w)
        return torch.stack([img_median, img_bilateral], dim=0), label

    def __len__(self):
        return (len(self.img_paths))


def data_generate(batch_size=1, num_of_letters=4, split=0.5, data_path=r'D:\learning-code\data\good', 
                  charsets='ABCDEFGHIJKLMNOPQRSTUVWXYZ', img_row=64, img_column=192, img_format='*.jpg'):
    num_of_classes = len(charsets)
    img_paths = glob.glob(os.path.join(data_path, img_format))
    random.shuffle(img_paths)
    basename = map(os.path.basename, img_paths)
    labels_letter_list = list(map(lambda name: name.split('.')[0].split('_')[0], basename))
    labels_index = [[charsets.index(letter) for letter in label_letter] for label_letter in labels_letter_list]
    labels = []
    for label_index in labels_index:
        label = torch.zeros(num_of_letters, num_of_classes).scatter_(1, torch.tensor(label_index).unsqueeze_(1), 1)
        labels.append(label)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((int(img_row), int(img_column)), antialias=True),
    ])

    captcha_ds_train = CaptchaDataset(img_paths, labels, transform, split)
    captcha_dl_train = DataLoader(captcha_ds_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    captcha_ds_test = CaptchaDataset(img_paths, labels, transform, split, train=False)
    captcha_dl_test = DataLoader(captcha_ds_test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return captcha_dl_train, captcha_dl_test


def img_generator(format=torch.Tensor, function=data_generate, batch_dim=True):
    _, test_dl = data_generate()
    for x, y in test_dl:
        if format == torch.Tensor:
            yield x if batch_dim else x[0]
        else:
            yield x.numpy().transpose((0,2,3,1)) if batch_dim else x.numpy().transpose((0,2,3,1))[0]


def probs_analysis(captcha_dl, charset=list(string.ascii_uppercase)):
    label_letter = None
    predicted_transition_matrix = torch.zeros([len(charset), len(charset)], dtype=torch.float64)
    for _, label in captcha_dl:
        indexes = torch.argmax(label, dim=-1) # (b, num_letters, num_class)  -> (b, um_letters)
        # 计算一阶统计量
        if label_letter is None:   
            label_letter = label.sum(axis=0)  # (b, num_letters, num_class) -> (num_letters, num_class) 
        else:
            label_letter += label.sum(axis=0) # Batch维度又可能不一样（dl中的最后一个batch),累加后消除了不一致的错误
    
        # 计算二阶Markov统计量
        for row in range(len(indexes)):
            for col in range(len(indexes[row])-1):
                predicted_transition_matrix[indexes[row][col], indexes[row][col+1]] += 1

    first_order_sta = label_letter / label_letter.sum(axis=-1, keepdims=True) # (num_letters, num_class) 
    predicted_transition_matrix /= predicted_transition_matrix.sum(axis=1, keepdims=True)
    return first_order_sta, predicted_transition_matrix 


def highest_lowest_chars(first_order_sta, highest_range=8, lowest_range=5, highest_num=2, lowest_num=2):
    data = first_order_sta.numpy().argsort()
    highest_result = Counter(data[:,-highest_range:].reshape(-1)).most_common(highest_num)
    lowest_result = Counter(data[:,:lowest_range].reshape(-1)).most_common(lowest_num)
    highest_Chars = list(map(lambda x: string.ascii_uppercase[x[0]], highest_result))
    lowest_Chars = list(map(lambda x: string.ascii_uppercase[x[0]], lowest_result))
    return highest_Chars, lowest_Chars
        

if __name__ == "__main__":

    
    img_gen = img_generator(format=torch.Tensor, batch_dim=1)

    imgs = next(img_gen)

    print("hello")
    # for i in range(5):
    #     print(next(img_gen).shape)
    # captcha_dl_train, captcha_dl_test = data_generate()
    # print(len(captcha_dl_train), len(captcha_dl_test))
    # first_order_sta, second_order_sta = probs_analysis(captcha_dl_train)
    # print(first_order_sta)
    # highest_Chars, lowest_Chars = highest_lowest_chars(first_order_sta, highest_range=8, lowest_range=5)
    # print(highest_Chars, lowest_Chars)




    # for (x, y) in captcha_dl_train:
    #     print(x[0][0][:3,:3])
    #     break


    # print(second_order_sta)
    # print('---------------------------------------------------------------------------')
    # print(second_order_sta.sum(axis=-1))
    # print('---------------------------------------------------------------------------')


