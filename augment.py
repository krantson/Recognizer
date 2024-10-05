import torch
import time
import cv2
import random
import shutil
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import shuffle_images



def add_random_curves(image, num_curves_max=20, curve_width=2):
    """
    """
    b, c, h, w = image.size()
    new_image = image
    # 创建一个新的空白图像用于绘制曲线
    device = image.device
    torch.manual_seed(int(time.time()))
    num_curves = random.randint(0, num_curves_max)

    for i_curve in range(num_curves):
        curves_image = (torch.zeros_like(image)).to(device)
        color = (2 * torch.rand((b, c, 1)) - 1).to(device)
        h1, w1 = int(torch.randint(low=1, high=h-1, size=(1,))), int(torch.randint(low=1, high=w-1, size=(1,)))
        # length = torch.randint(low=1, high=int(torch.min(h-h1, w-w1)), size=(1,))
        h2, w2 = int(torch.randint(low=1, high=h-1, size=(1,))), int(torch.randint(low=1, high=w-1, size=(1,)))
        length = min(abs(h2-h1), abs(w2-w1))
        h2 = (h1 + length) if h2 >= h1 else (h1 - length)
        w2 = (w1 + length) if w2 >= w1 else (w1 - length)
        h_index = torch.linspace(h1, h2, steps=length+1).to(torch.int)
        w_index = torch.linspace(w1, w2, steps=length+1).to(torch.int)
        curves_image[:, :, h_index, w_index] = color
        image = curves_image + image
    new_image = torch.clip(image, max=1, min=-1)
    return new_image


@torch.no_grad()
def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8 
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2)) # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...] 
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum() # 归一化
    return kernel


@torch.no_grad()
def bilateralBlur(batch_img, ksize=3, sigmaColor=None, sigmaSpace=None):
    device = batch_img.device
    if sigmaSpace is None:
        sigmaSpace = 0.15 * ksize + 0.35  # 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    if sigmaColor is None:
        sigmaColor = sigmaSpace
    
    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect').to(device)
    
    # batch_img 的维度为 BxcxHxW, 因此要沿着第 二、三维度 unfold
    # patches.shape:  B x C x H x W x ksize x ksize
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patch_dim = patches.dim() # 6 
    # 求出像素亮度差
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
    # 根据像素亮度差，计算权重矩阵
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
    # 归一化权重矩阵
    weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)
    
    # 获取 gaussian kernel 并将其复制成和 weight_color 形状相同的 tensor
    weights_space = getGaussianKernel(ksize, sigmaSpace).to(device)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)
    
    # 两个权重矩阵相乘得到总的权重矩阵
    weights = weights_space * weights_color
    # 总权重矩阵的归一化参数
    weights_sum = weights.sum(dim=(-1, -2))
    # 加权平均
    weighted_filter = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    return weighted_filter
    

@torch.no_grad()
def gaussianBlur(batch_img, ksize=3, sigma=0):
    device = batch_img.device
    kernel = getGaussianKernel(ksize, sigma).to(device) # 生成权重
    B, C, H, W = batch_img.shape # C：图像通道数，group convolution 要用到
    # 生成 group convolution 的卷积核
    kernel = kernel.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)
    pad = (ksize - 1) // 2 # 保持卷积前后图像尺寸不变
    # mode=relfect 更适合计算边缘像素的权重
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')
    image_fitlered = F.conv2d(batch_img_pad, weight=kernel, bias=None, 
                           stride=1, padding=0, groups=C)
    return image_fitlered


@torch.no_grad()
def medianBlur(batch_img, ksize=3):
    device = batch_img.device
    assert ksize % 2 == 1, "Kernel size should be odd"

    # 生成均值滤波的卷积核
    kernel = (torch.ones((ksize, ksize), dtype=torch.float32) / (ksize * ksize)).to(device)
    B, C, H, W = batch_img.shape

    # 生成用于分组卷积的卷积核权重
    kernel = kernel.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)
    pad = (ksize - 1) // 2

    # 对输入图像进行填充, mode=reflect 保持原图边缘的纹理
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')

    # 进行分组卷积
    weighted_fitlered = F.conv2d(batch_img_pad, weight=kernel, bias=None,
                            stride=1, padding=0, groups=C)
    return weighted_fitlered



class MultiViewFilter(nn.Module):
    """
    MultiViewFilter
    """
    def __init__(self, types='gm', ksize=3):
        super(MultiViewFilter, self).__init__()
        self.types = types
        self.ksize = ksize
        self.filters = {'g': gaussianBlur, 'm': medianBlur, 'b': bilateralBlur}
        
    def forward(self, input: torch.Tensor):
        # filtertype = random.choice(self.types)
        filtertype = 'g'
        output = self.filters[filtertype](input, ksize=self.ksize)
        # print(self.filters[filtertype])
        return output


def add_gaussian_noise(image, mean=0.0, std_max=0.3):
    """
    向图像添加高斯噪声
    :param image: PyTorch张量, 形状为 (C, H, W), 其中C是通道数, H是高度, W是宽度
    :param mean: 高斯噪声的平均值
    :param std: 高斯噪声的标准差
    :return: 带有高斯噪声的图像
    """
    # 生成与图像同形状的高斯噪声
    device = image.device
    b,c,h,w = image.shape
    torch.manual_seed(int(time.time()))
    std = (torch.rand(b,c,h,w) * std_max).to(device)
    noise = torch.randn_like(image) * std + mean
    
    # 将噪声添加到原始图像上
    noisy_image = torch.clip(image + noise, max=1, min=-1)
    return noisy_image


class LearnableUniformDistribution(nn.Module):
    def __init__(self, low_init=0.0, high_init=1.0):
        super(LearnableUniformDistribution, self).__init__()
        self.low = nn.Parameter(torch.tensor(low_init))
        self.high = nn.Parameter(torch.tensor(high_init))

    def forward(self, size):
        return torch.distributions.Uniform(self.low, self.high).sample(size)

    def get_parameters(self):
        return self.low, self.high


def pca_color_augmentation(image, alpha_std=0.1):
    device = image.device
    # Image format: (C, H, W)
    # Convert image to numpy and flatten spatial dimensions
    img = image.permute(1, 2, 0).cpu().numpy()
    orig_shape = img.shape
    img = img.reshape(-1, 3)

    # Compute the mean and the covariance matrix
    mean = np.mean(img, axis=0)
    img_centered = img - mean
    cov = np.cov(img_centered, rowvar=False) # 行方向是数据，列方向是变量, cov 是计算 RGB 通道之间相互关系的一个统计量，而不是单独对每个像素的值进行分析。

    # Eigen decomposition, 捕捉的RGB三个通道的主成分关系，不是每个像素之间的，所以 eigvals:(3,), eigvecs:(3,3)
    eigvals, eigvecs = np.linalg.eigh(cov) # eigvecs 每一列是一个特征向量

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigvals)[::-1] # 
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Generate random alpha values
    alphas = np.random.normal(0, alpha_std, size=(3,))

    # Compute the color shifts
    delta = np.dot(eigvecs, alphas * eigvals)
    img_shifted = img_centered + delta

    # Reconstruct the image
    img_shifted += mean
    img_shifted = np.clip(img_shifted, 0, 255)
    img_shifted = img_shifted.reshape(orig_shape)

    # Convert back to torch tensor
    img_shifted_tensor = torch.from_numpy(img_shifted).permute(2, 0, 1).float().to(device)
    return img_shifted_tensor


class PCACS(nn.Module):
    """
    PCA颜色增强
    """
    def __init__ (self, alpha_std=0.1):
        super(PCACS, self).__init__()
        self.alpha_std = alpha_std

    def forward(self, batch_images):
        # batch_images: Tensor with shape (B, C, H, W)
        augmented_images = []
        for image in batch_images:
            augmented_image = pca_color_augmentation(image, self.alpha_std)
            augmented_images.append(augmented_image)
        return torch.stack(augmented_images)


class AdaptiveColorShift(nn.Module):
    """
    随机颜色增强
    """
    def __init__(self):
        super(AdaptiveColorShift, self).__init__()
        self.r_ratio = nn.Parameter(torch.ones(1), requires_grad=True)
        self.g_ratio = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b_ratio = nn.Parameter(torch.ones(1), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input: torch.Tensor):
        device = input.device
        if len(input.shape) == 4:
            scale_shape = (1, 3, 1, 1)
            r1, g1, b1 = input[:,0,:,:].unsqueeze(1), input[:,1,:,:].unsqueeze(1), input[:,2,:,:].unsqueeze(1)
        else:
            r1, g1, b1 = input[0,:,:].unsqueeze(1), input[1,:,:].unsqueeze(1), input[2,:,:].unsqueeze(1)
            scale_shape = (3, 1, 1)
        # uniform random values
        one = torch.tensor(1).to(device)
        self.r = self.sigmoid(self.r_ratio).to(device)
        self.g = self.sigmoid(self.g_ratio).to(device)
        self.b = self.sigmoid(self.b_ratio).to(device)

        r_weight = torch.distributions.Uniform(one-self.r, one+self.r).sample(torch.Size([1]))[0]
        g_weight = torch.distributions.Uniform(one-self.g, one+self.g).sample(torch.Size([1]))[0]
        b_weight = torch.distributions.Uniform(one-self.b, one+self.b).sample(torch.Size([1]))[0]
        

        scale = torch.tensor([r_weight, g_weight, b_weight]).view(*scale_shape).to(device)
        output1 = input * scale
        return output1


class DCVCS(nn.Module):
    """
    空洞卷积变分颜色偏移 Dilated Convolutional-based Varaitional Color Shift
    """
    def __init__(self, kernel_size=8, p=0.0, shuffle=0, dim=3):
        super(DCVCS, self).__init__()
        if kernel_size == 8:
            dilation, stride, padding = (3, 9),  (22, 64), (1, 0)
        elif kernel_size == 4:
            dilation, stride, padding = (7, 21), (22, 64), (1, 0)
        elif kernel_size == 22:
            dilation, stride, padding = (1, 3),  (22, 64), (1, 0)

        self.shuffle = shuffle
        self.sigmoid = nn.Sigmoid()
        self.conv_upper = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv_lower = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(p=0.0)
        #self.conv_lower = nn.Conv2d(dim, dim, kernel_size=8, stride=64, dilation=9)

        self.conv_upper_weight = nn.Conv2d(dim, dim, (3,3), 1, 0)
        self.conv_lower_weight = nn.Conv2d(dim, dim, (3,3), 1, 0)
        # self.conv_upper_weight = nn.Conv2d(dim, dim, (1,3), 1, 0)
        # self.conv_lower_weight = nn.Conv2d(dim, dim, (1,3), 1, 0)
    
    def reparameterize(self, input, device):
        if self.shuffle:
            input_temp = shuffle_images(input)
            # print("shuffle")
        else:
            input_temp = input

        input_upper = self.conv_upper(input)
        input_lower = self.conv_lower(input)
        input_upper = self.dropout(input_upper)
        input_lower = self.dropout(input_lower)
        weight_upper = self.sigmoid(self.conv_upper_weight(input_upper)) 
        weight_lower = self.sigmoid(self.conv_lower_weight(input_lower)) 
        one = torch.ones_like(weight_upper).to(device)
        upper_limits = (one + weight_upper).expand_as(input)
        lower_limits = (one - weight_lower).expand_as(input)
        eps = torch.rand_like(input) # 从标准均匀分布中采用
        shift = lower_limits + (upper_limits - lower_limits) * eps
        ones = torch.ones_like(input).to(device)
        # return input + (ones - input)*(shift > 1.0)*(shift-ones) - input*(shift <=1.0)*(ones-shift) 
        return input * shift
    
    def forward(self, x):
        device = x.device
        return self.reparameterize(x, device)


# class DVCS(nn.Module):
#     """
#     变分颜色偏移 VariationalColorShift
#     """
#     def __init__(self, dim=3):
#         super(DVCS, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.dconv_upper = nn.Conv2d(dim, dim, (64, 192), 1, 0, groups=3)
#         self.pconv_upper = nn.Conv2d(dim, dim, 1, 1, 0)
#         self.dconv_lower = nn.Conv2d(dim, dim, (64, 192), 1, 0, groups=3)
#         self.pconv_lower = nn.Conv2d(dim, dim, 1, 1, 0)
    
#     def reparameterize(self, input, upper, lower, device):
#         weight_upper = self.sigmoid(upper)  
#         weight_lower = self.sigmoid(lower)
#         one = torch.ones_like(weight_upper).to(device)
#         upper_limits = (one + weight_upper).expand_as(input)
#         lower_limits = (one - weight_lower).expand_as(input)
#         eps = torch.rand_like(input) # 从标准均匀分布中采用
#         shift = lower_limits + (upper_limits - lower_limits) * eps
#         ones = torch.ones_like(input).to(device)
#         return shift * input

#     def forward(self, x):
#         device = x.device
#         upper = self.dconv_upper(x)
#         upper = self.pconv_upper(upper)
#         lower = self.dconv_lower(x)
#         lower = self.pconv_lower(lower)
#         return self.reparameterize(x, upper, lower, device)


class VCS(nn.Module):
    """
    变分颜色偏移 VariationalColorShift
    """
    def __init__(self, dim=3):
        super(VCS, self).__init__()
        self.ap2d = nn.AdaptiveAvgPool2d((1,3))
        self.mp2d = nn.AdaptiveMaxPool2d((1,3))
        self.sigmoid = nn.Sigmoid()
        self.conv_upper = nn.Conv2d(dim, dim, 3, 1, 0)
        self.conv_lower = nn.Conv2d(dim, dim, 3, 1, 0)

    
    def reparameterize(self, input, avg_input, max_input, min_input, device):
        mix_input = torch.concat([avg_input, max_input, min_input], dim=2)
        weight_upper = self.sigmoid(self.conv_upper(mix_input))  
        weight_lower = self.sigmoid(self.conv_lower(mix_input))
        one = torch.ones_like(weight_upper).to(device)
        upper_limits = (one + weight_upper).expand_as(input)
        lower_limits = (one - weight_lower).expand_as(input)
        eps = torch.rand_like(input) # 从标准均匀分布中采用
        shift = lower_limits + (upper_limits - lower_limits) * eps
        return input * shift

    def forward(self, x):
        device = x.device
        avg_input = self.ap2d(x)
        max_input = self.mp2d(x)
        min_input = -self.mp2d(-x)
        return self.reparameterize(x, avg_input, max_input, min_input, device)



class RCS(nn.Module):
    """
    随机颜色偏移 RadomColorShift
    """
    def __init__(self, ratio=0.2, dim=3):
        super(RCS, self).__init__()
        self.ratio = torch.tensor(float(ratio))
    
    def forward(self, input: torch.Tensor):
        device = input.device
        self.ratio = self.ratio.to(input)
        one = torch.ones_like(self.ratio).to(device)
        upper_limits = (one + self.ratio).expand_as(input)
        lower_limits = (one - self.ratio).expand_as(input)
        eps = torch.rand_like(input) # 从标准均匀分布中采用
        shift = lower_limits + (upper_limits - lower_limits) * eps
        return input * shift


class ACS(nn.Module):
    """
    自适应颜色偏移 AdaptiveColorShift
    """
    def __init__(self, dim=3):
        super(ACS, self).__init__()
        self.conv_ratio = nn.Conv2d(dim, dim, 3, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        device = input.device
        mix_input = input.new_ones([1, 3, 3, 3])
        ratio = self.sigmoid(self.conv_ratio(mix_input))  
        one = torch.ones_like(ratio).to(device)
        upper_limits = (one + ratio).expand_as(input)
        lower_limits = (one - ratio).expand_as(input)
        eps = torch.rand_like(input) # 从标准均匀分布中采用
        shift = lower_limits + (upper_limits - lower_limits) * eps
        return input * shift



class MixChannel(nn.Module):
    """
    随机颜色增强
    """
    def __init__(self):
        super(MixChannel, self).__init__()
        self.r_ratio = nn.Parameter(torch.ones(1), requires_grad=True)
        self.g_ratio = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b_ratio = nn.Parameter(torch.ones(1), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input: torch.Tensor):
        device = input.device
        if len(input.shape) == 4:
            scale_shape = (1, 3, 1, 1)
            r1, g1, b1 = input[:,0,:,:].unsqueeze(1), input[:,1,:,:].unsqueeze(1), input[:,2,:,:].unsqueeze(1)
        else:
            r1, g1, b1 = input[0,:,:].unsqueeze(1), input[1,:,:].unsqueeze(1), input[2,:,:].unsqueeze(1)
            scale_shape = (3, 1, 1)
        # uniform random values
        one = torch.tensor(2).to(device)
        self.r = self.sigmoid(self.r_ratio).to(device) 
        self.g = self.sigmoid(self.g_ratio).to(device)
        self.b = self.sigmoid(self.b_ratio).to(device)

        scale = torch.tensor([self.r, self.g, self.b]).view(*scale_shape).to(device)
        output1 = input * scale
        return output1







if __name__ == '__main__':
    pass
    # from dataset import *
    # img_gen = img_generator(format=torch.Tensor, batch_dim=1)
    # image = next(img_gen)
    # acs = AdaptiveColorShift()
    # image = cv2.imread('223.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image.astype(np.float32) / 255.0
    # image = np.transpose(image, (2, 0, 1))
    # tensor_image = torch.from_numpy(image)
    # new_image = acs(tensor_image.unsqueeze(0))
    


    # new_image = add_gaussian_noise(image)
    # # new_image = add_random_curves(image)
    # fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    # ax[0].imshow(image.squeeze(0).permute(1, 2, 0).numpy())
    # ax[0].set_title("Original Image")
    # ax[0].axis('off')  # 不显示坐标轴

    # 显示加噪声后的图像
    # ax.imshow(new_image.squeeze(0).permute(1, 2, 0).numpy())
    # ax.set_title("Noisy Image")
    # ax.axis('off')  # 不显示坐标轴
    # fig.savefig(r'D:\new.jpg', dpi=100, bbox_inches='tight')
    # plt.show()




