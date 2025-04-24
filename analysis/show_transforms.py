import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

transform_crop = transforms.RandomCrop(28, padding=4)
transform_flip = transforms.RandomHorizontalFlip(p=1.0)  # 为了确定性，设 p=1.0 总是翻转

img = Image.open("./data/images/8_2576.png").convert("L")  

imgs = [] # 保存数据处理各个阶段的图片

imgs.append(img) # 原图
img_cropped = transform_crop(img)
imgs.append(img_cropped)
img_flipped = transform_flip(img_cropped)
imgs.append(img_flipped)

fig, axes = plt.subplots(1, len(imgs)*2-1, figsize=(20,5))

for i in range(len(axes)):
    if i % 2 == 0:
        img_idx = i//2
        axes[i].imshow(imgs[img_idx], cmap='gray')
        axes[i].axis('off')
    else:
        axes[i].text(0.5, 0.5, '→', fontsize=60, ha='center', va='center')
        axes[i].axis('off')

plt.tight_layout()
plt.show()
