from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from collections import Counter
from functools import partial
from utils import *
import matplotlib.pyplot as plt
import shutil
import string
import uuid
import glob
import cv2
import os
import torch
import random
import numpy as np


random.seed(0)
torch.manual_seed(0)


class CaptchaDataset(Dataset):
    def __init__(self, img_paths, labels, transform, colormode='gray'):
        super().__init__()
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.colormode = colormode
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        img = cv2.imread(img_path)  # (h, w, c), BGR, ToTensor 不转换颜色通道
 
        # 默认保持BGR
        if self.colormode.upper() == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # (h, w, c)->(h, w)
        elif self.colormode.upper() == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     

        img = self.transform(img) # (h, w) -> (1, h, w)
        return img, label

    def __len__(self):
        return (len(self.img_paths))


def get_labels_from_img_paths(img_paths, charsets='ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    basenames = map(os.path.basename, img_paths)
    labels_letter_list = list(map(lambda name: name.split('.')[0].split('_')[0], basenames))
    labels_index = [[charsets.index(letter) for letter in label_letter] for label_letter in labels_letter_list]
    labels = []
    num_of_classes = len(charsets)
    for label_index in labels_index:
        label = torch.zeros(len(label_index), num_of_classes).scatter_(1, torch.tensor(label_index).unsqueeze_(1), 1)
        labels.append(label)
    return labels


def data_generate(batch_size=1, num_of_letters=4, split=0.5, data_path=r'D:\learning-code\data\good', 
                  charsets='ABCDEFGHIJKLMNOPQRSTUVWXYZ', img_row=64, img_column=192, 
                  img_format="*.[jp][pn]g", colormode='gray'):
    img_format = (f"**/{img_format}" if '*' in img_format else f"**/*{img_format}") if img_format else "**/*.[jp][pn]g"
 
    if type(data_path) == str:
        img_paths = glob.glob(os.path.join(data_path, img_format), recursive=True)
        random.shuffle(img_paths)

    # check train and (val or valid) directory
    train_path = os.path.join(data_path, 'train') if type(data_path) == str and os.path.isdir(os.path.join(data_path, 'train')) else None
    if type(data_path) == str and os.path.isdir(os.path.join(data_path, 'val')):
        val_path = os.path.join(data_path, 'val')
    elif type(data_path) == str and os.path.isdir(os.path.join(data_path, 'valid')):
        val_path = os.path.join(data_path, 'valid')
    else:
        val_path = None

    # split dataset
    if train_path and val_path:
        print("Train and Val directory found!")
        train_img_paths = list(filter(lambda x: train_path in x, img_paths))
        val_img_paths = list(filter(lambda x: val_path in x, img_paths))
    elif type(data_path) == str and (type(split) == float or type(split) == int) and 1 >= split > 0:
        print(f"No train and val directory, use split {split}")
        train_img_paths = img_paths[:int(len(img_paths) * split)]
        val_img_paths = img_paths[int(len(img_paths) * split):]
    elif type(data_path) != str:
        train_img_paths, val_img_paths = [], []
        for dir_path in data_path:
            img_paths = glob.glob(os.path.join(dir_path, img_format), recursive=True)
            random.shuffle(img_paths)
            if 'val' not in dir_path.lower():
                train_img_paths.extend(img_paths)
            else:
                val_img_paths.extend(img_paths)
    else:
        raise FileNotFoundError
    
    train_labels = get_labels_from_img_paths(train_img_paths, charsets)
    val_labels = get_labels_from_img_paths(val_img_paths, charsets)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((int(img_row), int(img_column)), antialias=True),
    ])

    # get loader
    captcha_ds_train = CaptchaDataset(train_img_paths, train_labels, transform, colormode)
    captcha_dl_train = DataLoader(captcha_ds_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    captcha_ds_test = CaptchaDataset(val_img_paths, val_labels, transform, colormode)
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


def symmetry_augmentation(src_path, dst_path, colormode='rgb'):

    h_dst_path = os.path.join(dst_path, "H_Symmetry")
    v_dst_path = os.path.join(dst_path, "V_Symmetry")
    d_dst_path = os.path.join(dst_path, "D_Symmetry")
    o_dst_path = os.path.join(dst_path, "Original")
    dst_path_dict = {'H':h_dst_path, 'V': v_dst_path, 'D': d_dst_path, 'O': o_dst_path}

    for d_path in [h_dst_path, v_dst_path, d_dst_path, o_dst_path]:
        if os.path.exists(d_path):
            shutil.rmtree(d_path)
        os.makedirs(d_path, exist_ok=True)
    
    dl_train, _ = data_generate(batch_size=1, data_path=src_path, split=1, colormode=colormode)

    symmetry_index_dict = {}
    symmetry_flip_dict = {}
    # horizatoal flip
    h_symmetry = ['A', 'H', 'I', 'M', 'O', 'T', 'U', 'V', 'W', 'X', 'Y']
    h_symmetry_index = torch.tensor(list(map(lambda x: string.ascii_uppercase.index(x), h_symmetry)))
    symmetry_index_dict['H'] = h_symmetry_index
    symmetry_flip_dict['H'] = partial(torch.flip, dims=[-1])
    # vertical flip
    v_symmetry = ['B', 'C', 'D', 'E', 'H', 'I', 'K', 'O', 'X']
    v_symmetry_index = torch.tensor(list(map(lambda x: string.ascii_uppercase.index(x), v_symmetry)))
    symmetry_index_dict['V'] = v_symmetry_index
    symmetry_flip_dict['V'] = partial(torch.flip, dims=[-2])
    # 180 degree flip
    v_symmetry = ['H', 'I', 'N', 'O', 'S', 'X', 'Z']
    v_symmetry_index = torch.tensor(list(map(lambda x: string.ascii_uppercase.index(x), v_symmetry)))
    symmetry_index_dict['D'] = v_symmetry_index
    symmetry_flip_dict['D'] = partial(torch.flip, dims=[-2, -1])

    for img, label in dl_train:

        m = torch.nonzero(label)[::,-1].reshape(label.shape[0], -1)
        flag = False

        for symmetry_mode in ['H', "V", "D"]: # H:horizontal, V:vertical, D: diagonal

            symmetry_index = symmetry_index_dict[symmetry_mode]
            flip = symmetry_flip_dict[symmetry_mode]
            save_path = dst_path_dict[symmetry_mode]

            if torch.sum(m[..., None] == symmetry_index, dim=(-2, -1)) == 4:
                img_ = flip(img)
                save_image(img_, os.path.join(save_path, onehot_to_alphabet(label[0]) + '_' + str(uuid.uuid4())) + '_' + symmetry_mode + '.jpg')
                flag = True
                print(f"Saved img to {save_path}.")
        
        if flag:
            save_image(img, os.path.join(dst_path_dict['O'], onehot_to_alphabet(label[0]) + '_' + str(uuid.uuid4())  + '.jpg'))


        

if __name__ == "__main__":

    # dl_train, dl_test = data_generate(batch_size=1, data_path=r'D:\learning-code\data\sample', split=1)

    # print(len(iter(dl_train)))

    symmetry_augmentation(src_path=r"D:\learning-code\data\syncaptcha13000\mcaptcha2_5000", 
                         dst_path=r"D:\learning-code\data\syncaptcha13000",
                         colormode="rgb")

    # img_gen = img_generator(format=torch.Tensor, batch_dim=1)
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


