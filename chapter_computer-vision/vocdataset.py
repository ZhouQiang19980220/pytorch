"""
读取VOC语义分割数据集
"""

#%%
import unittest
import random
from loguru import logger
import os
import torch
from torch import nn
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

#%%
# 这里将全部的图片读取到内存中, 是因为VOC数据集比较小
# 如果数据集比较大, 可以使用生成器, 每次读取一个batch的数据
def get_seg_voc_dataset(voc_root: str, is_train: bool, batch_size: int=16):
    """
    逐批量读取 VOC 分割数据集
    """
    txt_fname = os.path.join(voc_root, 'ImageSets/Segmentation/', 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f: # 获取所有图片的文件名(无后缀)
        images = f.read().split()   
    logger.debug(f'Number of images: {len(images)}')
    logger.debug(f'First 5 images: {images[:5]}')

    img_dir = os.path.join(voc_root, 'JPEGImages')
    label_dir = os.path.join(voc_root, 'SegmentationClass')


    for i in range(0, len(images), batch_size):
        start = i
        if i + batch_size > len(images):
            end = len(images)
        else:
            end = i + batch_size
        batch_images = images[start:end]
        batch_features = []
        batch_labels = []
        for fname in batch_images:
            img = os.path.join(img_dir, f'{fname}.jpg')
            label = os.path.join(label_dir, f'{fname}.png')
            batch_features.append(Image.open(img).convert('RGB'))
            batch_labels.append(Image.open(label).convert('RGB'))
        yield batch_features, batch_labels

class MyVOCSegDataset(torch.utils.data.Dataset):
    """
    自定义的 VOC 数据集
    """
    VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

    #@save
    VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

    def __init__(
        self, 
        voc_dir: str,
        is_train: bool = True,
        transform = None, 
    ):
        self.voc_dir = voc_dir
        self.img_dir = os.path.join(voc_dir, 'JPEGImages')
        self.label_dir = os.path.join(voc_dir, 'SegmentationClass')
        self.transform = transform
        
        txt_fname = os.path.join(voc_dir, 'ImageSets/Segmentation/', 'train.txt' if is_train else 'val.txt')

        with open(txt_fname, 'r') as f:
            self.images = f.read().split()
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        img = os.path.join(self.img_dir, f'{fname}.jpg')
        label = os.path.join(self.label_dir, f'{fname}.png')
        img = Image.open(img).convert('RGB')
        label = Image.open(label).convert('RGB')
        
        if self.transform:
            img, label = self.transform((img, label))
        label = self.voc_label_indices(label, self.voc_colormap2label)

        
        return img, label

    @property
    def voc_colormap2label(self, ):
        """Build the mapping from RGB to class indices for VOC labels."""
        VOC_COLORMAP = self.VOC_COLORMAP
        colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
        for i, colormap in enumerate(VOC_COLORMAP):
            colormap2label[
                (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
        return colormap2label

    def voc_label_indices(self, colormap, colormap2label):
        """Map any RGB values in VOC labels to their class indices."""
        colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
        # idx.shape = (height, width)
        idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
            + colormap[:, :, 2])
        # print(f'idx.shape = {idx.shape}')
        # print(f'colormap.shape = {colormap.shape}')
        return colormap2label[idx]
            
# 测试自定义的数据集
class TestMyVOCSegDataset(unittest.TestCase):
    def test_my_voc_seg_dataset(self):
        voc_dir = '../data/VOCdevkit/VOC2012/'
        voc_dataset = MyVOCSegDataset(voc_dir)
        img, label = voc_dataset[0]
        self.assertEqual(img.size, label.size)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 

#%% 定义用于分割任务的 trans
class SegTransform:
    """
    定义用于分割任务的图像增强
    关键在于对 image 和 label 应用参数相同的 transform
    """

    def __init__(self, trans, only_image=False):
        self.trans = trans  
        self.only_image = only_image


    def __call__(self, sample_tuple):
        img, label = sample_tuple
        seed = torch.randint(0, 2**32-1, (1, )).item()
        set_seed(seed)

        img = self.trans(img)
        if not self.only_image:
            set_seed(seed)  # 确保 label 使用相同的随机种子
            label = self.trans(label)

        return img, label

#%%
def main():
    # interactive mode
    plt.ion()
    # 定义几何变换，同时应用于 image 和 label
    geo_trans = transforms.Compose([
        transforms.RandomResizedCrop(size=224), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15)
    ])

    # 定义颜色变换，单独应用于 image
    color_trans = transforms.Compose([
        # 分别表示亮度、对比度、饱和度、色调的变化范围
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.01),
    ])

    # 定义转换

    training_trans = transforms.Compose([
        SegTransform(geo_trans, only_image=False),
        SegTransform(color_trans, only_image=True),
        SegTransform(transforms.ToTensor(), only_image=False),
    ])



    voc_dir = 'data/VOCdevkit/VOC2012/'
    training_dataset = MyVOCSegDataset(voc_dir, is_train=True, transform=training_trans)

    # 测试数据集
    img, label = training_dataset[0]
    plt.subplot(121)
    plt.imshow(img.permute(1, 2, 0))
    plt.subplot(122)
    plt.imshow(label.permute(1, 2, 0))
    # 关闭坐标轴
    plt.axis('off')
    # test_dataset = MyVOCSegDataset(voc_dir, is_train=False)


#%%
if __name__ == '__main__':
    # unittest.main() 
    main()

#%%
