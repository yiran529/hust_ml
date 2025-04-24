import gzip
import numpy as np
import os
from PIL import Image

def load_fashion_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)

def load_fashion_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)

images = load_fashion_mnist_images('.\\data\\FashionMNIST\\raw\\t10k-images-idx3-ubyte.gz')
labels = load_fashion_mnist_labels('.\\data\\FashionMNIST\\raw\\t10k-labels-idx1-ubyte.gz')

output_dir = './data/images'
os.makedirs(output_dir, exist_ok=True)

for i, (img, label) in enumerate(zip(images, labels)):
    Image.fromarray(img, mode='L').save(f'{output_dir}/{label}_{i}.png')

print(f"已保存 {len(images)} 张图片到目录: {output_dir}")