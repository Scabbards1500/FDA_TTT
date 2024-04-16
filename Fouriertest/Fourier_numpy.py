import cv2
import numpy as np
import matplotlib.pyplot as plt


filepath = r"D:\tempdataset\TTADataset\CHASE\test\images512\Image_01L.jpg"
# 读取图像
# image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
image = cv2.imread(filepath)[:,:,::-1] #cv2默认是BGR通道顺序，这里调整到RGB
img = cv2.resize(image,(512,512))

fre = np.fft.fft2(img,axes=(0,1)) #变换得到的频域图数据是复数组成的
fre_m = np.abs(fre)   #幅度谱，求模得到
fre_p = np.angle(fre) #相位谱，求相角得到

# 把相位设为常数
constant = fre_m.mean()
fre_ = constant * np.e ** (1j * fre_p)  # 把幅度谱和相位谱再合并为复数形式的频域图数据
img_onlyphase = np.abs(np.fft.ifft2(fre_, axes=(0, 1)))  # 还原为空间域图像

#把振幅设为常数
constant = fre_p.mean()
fre_ = fre_m * np.e ** (1j * constant)
img_onlymagnitude = np.abs(np.fft.ifft2(fre_, axes=(0, 1)))

constant = fre_p
fre_ = fre_m * np.e ** (1j * constant)
img_ori = np.abs(np.fft.ifft2(fre_, axes=(0, 1)))

# 绘制原始图像和振幅谱
plt.subplot(1, 3, 1), plt.imshow(img.astype('uint8'))
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(img_onlyphase.astype('uint8'))
plt.title('phase Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(img_onlymagnitude.astype('uint8'))
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()






