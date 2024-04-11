import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms



import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
import cv2

# 读取图像
# 图像转tensor！
# 读取图像
image1 = Image.open(r"D:\tempdataset\TTADataset\CHASE\test\images512\Image_01L.jpg")  # 替换为你的图像文件路径
image2 = Image.open(r"D:\tempdataset\HRF\images512\01dr.JPG")
# 定义转换
transform = transforms.ToTensor()
# 将图像转换为张量
tensor1 = transform(image1)
tensor2 = transform(image2)


def fourier(tensor):
    # 进行傅里叶变换
    fre = torch.fft.fftn(tensor, dim=(-2, -1))  # 在图像的最后两个维度上执行傅里叶变换
    fre_m = torch.abs(fre)   # 幅度谱，求模得到
    fre_p = torch.angle(fre) # 相位谱，求相角得到
    return fre_m, fre_p


fre_m1, fre_p1 = fourier(tensor1)
fre_m2, fre_p2 = fourier(tensor2)

    # 使用逆傅立叶变换获取时域信号
# fre_img = fre_m1 * torch.exp(1j * fre_p1)
# img = torch.abs(torch.fft.ifftn(fre_img, dim=(-2, -1)))


# 把相位设为常数
constant = torch.mean(fre_m1)
fre_ = constant * torch.exp(1j * fre_p1)  # 把幅度谱和相位谱再合并为复数形式的频域图数据
img_onlyphase = torch.abs(torch.fft.ifftn(fre_, dim=(-2, -1)))  # 还原为空间域图像

# 把振幅设为常数
constant = torch.mean(fre_p1)
fre_ = fre_m1 * torch.exp(1j * constant)
img_onlymagnitude = torch.abs(torch.fft.ifftn(fre_, dim=(-2, -1)))


# tensor转图像！
# 定义转换
transform = transforms.ToPILImage()
# 将张量转换为图像
image2 = transform(img_onlymagnitude)
# 显示图像
plt.imshow(image2)
plt.axis('off')  # 不显示坐标轴
plt.show()

