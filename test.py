import glob
import os
import cv2
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from torchvision.utils import make_grid
from models import Model, load_model_nopara
from utils import str_to_model_kwargs, str_to_paras, RGB2gray, calc_psnr, calc_ssim


def show_module_output(data_path='train', model_file_path='', img_format="*.jpg", seed=None, num_figs=1, only_output=True, ssim=False):
    model = load_model_nopara(model_file_path)
    checkpoint = torch.load(model_file_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])

    if seed != None:
        random.seed(seed)
    toTensor = transforms.ToTensor()
    img_paths = glob.glob(os.path.join(data_path, img_format))
    random.shuffle(img_paths)

    img_inputs, img_outputs = [], []
    for img_path in img_paths[:num_figs]:
        # img_np = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        img_np = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # img_np = cv2.imread(img_path)
        img = toTensor(img_np)
        # img_np = img.squeeze().detach().numpy()
        img_inputs.append(img)
        img_out = model.feature_output(img).squeeze(0)
        img_out_np = img_out.squeeze().detach().numpy()
        # img_out = (img_out - torch.min(img_out)) / (torch.max(img_out) - torch.min(img_out))
        img_outputs.append(img_out)
        if ssim:
            img_np_norm = np.zeros(img_np.shape, dtype=np.float32)
            img_out_np_norm = np.zeros(img_out_np.shape, dtype=np.float32) 
            cv2.normalize(img_np, img_np_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.normalize(img_out_np, img_out_np_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            print(ssim(img_np_norm, img_out_np_norm))

    all_imgs = img_outputs if only_output else (img_inputs + img_outputs)
    padding = 0 if num_figs == 1 else 10
    if len(all_imgs) == 1:
        imgs_np = np.transpose(img_out_np, (1,2,0))
    else:
        imgs = make_grid(all_imgs, nrow=num_figs, padding=padding, normalize=True, scale_each=True, pad_value=1)
        imgs_np = np.transpose(imgs.numpy(), (1,2,0))
    plt.imsave('image_filter_contrast.jpg', imgs_np, dpi=600)
    plt.imshow(imgs_np)
    plt.axis("off")
    plt.show()




def predict_with_model(data_path='train', model_file_path='', img_format="*.jpg", seed=None, num_figs=2):
    model = load_model_nopara(model_file_path)
    checkpoint = torch.load(model_file_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])
    if seed != None:
        random.seed(seed)
    toTensor = transforms.ToTensor()
    img_paths = glob.glob(os.path.join(data_path, img_format))
    random.shuffle(img_paths)

    img_inputs, img_outputs = [], []
    for img_path in img_paths[:num_figs]:
        img = toTensor(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY))
        model.eval()
        model.feature_output(img)


def noise_analysis(data_path='train', model_file_path='', img_format="*.jpg", seed=None, num_figs=3):
    model = load_model_nopara(model_file_path)
    checkpoint = torch.load(model_file_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])

    if seed != None:
        random.seed(seed)
    toTensor = transforms.ToTensor()
    img_paths = glob.glob(os.path.join(data_path, img_format))
    random.shuffle(img_paths)

    img_inputs, img_outputs = [], []
    plt.figure(figsize=(12, 6))
    for img_path in img_paths[:num_figs]:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        img_tensor = toTensor(img)
        fshift = np.fft.fftshift(np.fft.fft2(img))
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        magnitude_spectrum_tensor = toTensor(magnitude_spectrum)
        img_inputs.append(magnitude_spectrum_tensor)
        img_tensor_out = model.feature_output(img_tensor).squeeze(0)
        img_out = np.array(img_tensor_out.detach()).transpose(1,2,0)
        fshift = np.fft.fftshift(np.fft.fft2(img_out))
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        magnitude_spectrum_tensor = toTensor(magnitude_spectrum)
        img_outputs.append(magnitude_spectrum_tensor)

    imgs = make_grid(img_inputs + img_outputs, nrow=num_figs, padding=10, normalize=True, pad_value=1)
    imgs = np.transpose(imgs.numpy(), (1,2,0))
    plt.imsave('image_spectrum_contrast.jpg', imgs, dpi=600)
    plt.imshow(imgs)
    plt.axis("off")
    plt.show()


def test_ssim_psnr(original_data_path, target_data_path, model_file_path='',  seed=None, num_figs=3):
    model = load_model_nopara(model_file_path)
    checkpoint = torch.load(model_file_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])

    if seed != None:
        random.seed(seed)

    toTensor = transforms.ToTensor()
    original_paths = glob.glob(os.path.join(original_data_path, '*.png')) + glob.glob(os.path.join(original_data_path, '*.jpg'))
    target_paths = glob.glob(os.path.join(target_data_path, '*.png')) + glob.glob(os.path.join(target_data_path, '*.jpg'))
    cumulative_psnr = 0
    cumulative_ssim = 0
    assert len(original_paths) == len(target_paths)
    dataset_len = len(original_paths)
    for original_path, target_path in zip(original_paths[:num_figs], target_paths[:num_figs]):
        assert os.path.basename(original_path)[:-4] == os.path.basename(target_path)[:-4]
        original_img_np = cv2.cvtColor(cv2.imread(original_path), cv2.COLOR_BGR2RGB)
        original_img = toTensor(original_img_np)
        target_img_np = cv2.cvtColor(cv2.imread(target_path), cv2.COLOR_BGR2RGB)
        target_img = toTensor(target_img_np)
        original_img_out = model.feature_output(original_img)
        target_img_unsqueeze = target_img.unsqueeze(0)
        # PSNR
        cur_psnr = calc_psnr(target_img_unsqueeze, original_img_out)
        # SSIM
        original_output_gray, target_gray = RGB2gray(original_img_out, keepdim=True), RGB2gray(target_img_unsqueeze, keepdim=True)
        cur_ssim = calc_ssim(original_output_gray[0], target_gray[0])
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim
    psnr_per_image = cumulative_psnr / dataset_len
    ssim_per_image = cumulative_ssim / dataset_len
    print('psnr_per_image:', psnr_per_image, 'sim_per_image:', ssim_per_image)



if __name__ == "__main__":
    show_module_output(data_path=r"D:\my-code\data\dataset\1\denoise\noisy16", 
    model_file_path=r"D:\Results\FEGAN result\preprocessing_compare\mcaptcha\新建文件夹\deepcaptcha_aug@P(DEEPCAPTCHA='CNN', AUG='multiviewfilter_gb')#M-CAPTCHA-8000.pth")
    # test_ssim_psnr(original_data_path=r'D:\my-code\data\dataset\1\train_clean_100',
    # target_data_path=r'D:\my-code\data\dataset\1\train_noisy_100',
    # model_file_path=r"D:\Results\FEGAN result\preprocessing_compare\mcaptcha\新建文件夹\deepcaptcha_aug@P(DEEPCAPTCHA='CNN', AUG='multiviewfilter_gm')#M-CAPTCHA-8000.pth")




    


