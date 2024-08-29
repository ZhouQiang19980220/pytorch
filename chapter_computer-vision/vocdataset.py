# """
# 读取VOC语义分割数据集
# """

# #%%
# import unittest
# import random
# from loguru import logger
# import os
# import torch
# from torch import nn
# import torchvision
# from torchvision import transforms
# from matplotlib import pyplot as plt
# import numpy as np
# from PIL import Image

# #%%
# # 这里将全部的图片读取到内存中, 是因为VOC数据集比较小
# # 如果数据集比较大, 可以使用生成器, 每次读取一个batch的数据
# def get_seg_voc_dataset(voc_root: str, is_train: bool, batch_size: int=16):
#     """
#     逐批量读取 VOC 分割数据集
#     """
#     txt_fname = os.path.join(voc_root, 'ImageSets/Segmentation/', 'train.txt' if is_train else 'val.txt')
#     with open(txt_fname, 'r') as f: # 获取所有图片的文件名(无后缀)
#         images = f.read().split()   
#     logger.debug(f'Number of images: {len(images)}')
#     logger.debug(f'First 5 images: {images[:5]}')

#     img_dir = os.path.join(voc_root, 'JPEGImages')
#     label_dir = os.path.join(voc_root, 'SegmentationClass')


#     for i in range(0, len(images), batch_size):
#         start = i
#         if i + batch_size > len(images):
#             end = len(images)
#         else:
#             end = i + batch_size
#         batch_images = images[start:end]
#         batch_features = []
#         batch_labels = []
#         for fname in batch_images:
#             img = os.path.join(img_dir, f'{fname}.jpg')
#             label = os.path.join(label_dir, f'{fname}.png')
#             # batch_features.append(Image.open(img).convert('RGB'))
#             # batch_labels.append(Image.open(label).convert('RGB'))
#             batch_features.append(Image.open(img))
#             batch_labels.append(Image.open(label))

#         yield batch_features, batch_labels

# #%%

# class MyVOCSegDataset(torch.utils.data.Dataset):
#     """
#     自定义的 VOC 数据集
#     """

#     def __init__(
#         self, 
#         voc_dir: str,
#         is_train: bool = True,
#         transform = None, 
#     ):
#         self.voc_dir = voc_dir
#         self.img_dir = os.path.join(voc_dir, 'JPEGImages')
#         self.label_dir = os.path.join(voc_dir, 'SegmentationClass')
#         self.transform = transform
        
#         txt_fname = os.path.join(voc_dir, 'ImageSets/Segmentation/', 'train.txt' if is_train else 'val.txt')

#         with open(txt_fname, 'r') as f:
#             self.images = f.read().split()

#         # 获取所有图片的文件名
#         self.img_fnames = [os.path.join(self.img_dir, f'{fname}.jpg') for fname in self.images]
#         self.label_fnames = [os.path.join(self.label_dir, f'{fname}.png') for fname in self.images]
    
#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         fname = self.images[idx]
#         img_fname = self.img_fnames[idx]
#         label_fanme = self.label_fnames[idx]
#         img: Image.Image = Image.open(img_fname)
#         label: Image.Image = Image.open(label_fanme)

#         if idx == 0:
#             logger.debug(f'{img.size=}, {label.size=}')
        
#         if self.transform:
#             img, label = self.transform((img, label))

#         return img, label
            
# # 测试自定义的数据集
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.random.manual_seed(seed)

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed) 

# #%% 定义用于分割任务的 trans
# class BaseSegTransform:
#     """
#     定义用于分割任务的图像增强
#     关键在于对 image 和 label 应用参数相同的 transform
#     """

#     def __init__(self, trans, only_image=False):
#         self.trans = trans  
#         self.only_image = only_image


#     def __call__(self, sample_tuple):
#         img, label = sample_tuple
#         seed = torch.randint(0, 2**32-1, (1, )).item()
#         set_seed(seed)

#         img = self.trans(img)
#         if not self.only_image:
#             set_seed(seed)  # 确保 label 使用相同的随机种子
#             label = self.trans(label)

#         return img, label

# class SegRandomResizedCrop():
#     """
#     随机裁剪
#     """
#     def __init__(self, size, scale=(0.5, 1.0), ratio=(0.75, 1.33)):
#         self.size = size
#         self.scale = scale
#         self.ratio = ratio

#     def __call__(self, sample_tuple):
#         img, label = sample_tuple
#         i, j, h, w = transforms.RandomResizedCrop.get_params(
#             img, scale=self.scale, ratio=self.ratio
#         )
#         img = transforms.functional.resized_crop(img, i, j, h, w, self.size)
#         # 注意这里使用最近邻插值, 为了保证 label 的dtype是 torch.int64
#         label = transforms.functional.resized_crop(label, i, j, h, w, self.size, interpolation=Image.NEAREST)
#         return img, label

# class SegToTensor():

#     def __call__(self, sample_tuple):
#         img, label = sample_tuple
#         img = transforms.functional.to_tensor(img)
#         label = torch.as_tensor(np.array(label), dtype=torch.int64)
#         return img, label

# #%%
# if __name__ == '__main__':
#     plt.ion()
#     BASE_DIR = os.path.dirname(
#         os.path.abspath(__file__)
#     )
#     ROOT_DIR = '/root/pytorch'
#     voc_dir = os.path.join(ROOT_DIR, 'data/VOCdevkit/VOC2012/')

#     # 测试自定义的数据集
#     # dataset = MyVOCSegDataset(voc_dir, is_train=True, transform=SegToTensor())
#     # dataset = MyVOCSegDataset(voc_dir, is_train=True)
#     # img, label = dataset[0]

#     to_pil = transforms.ToPILImage()
#     # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#     # for ax in axes:
#     #     ax.axis('off')
#     # axes[0].imshow(to_pil(img))
#     # axes[1].imshow(np.array(label))

#     train_transforms = [
#         SegRandomResizedCrop((224, 224)),
#         BaseSegTransform(transforms.RandomHorizontalFlip(), only_image=False),
#         BaseSegTransform(transforms.RandomVerticalFlip(), only_image=False),
#         BaseSegTransform(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), only_image=True),
#         SegToTensor(), 
#         BaseSegTransform(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), only_image=True),
#     ]
#     train_transforms = transforms.Compose(train_transforms)
#     dataset = MyVOCSegDataset(voc_dir, is_train=True, transform=train_transforms)
#     fig, axes = plt.subplots(3, 3, figsize=(12, 12))
#     axes = axes.flatten()
#     for i in range(9):
#         img, label = dataset[i]
#         axes[i].imshow(to_pil(img))
#         axes[i].imshow(np.array(label), alpha=0.5)
#         axes[i].axis('off')

# #%%

#%% import libraries
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
import random

#%% define dataset
class MyVOCSegDataset(Dataset):

    def __init__(self, voc_dir: str, is_train: bool=True, transform=None):
        super().__init__()
        self.voc_dir = voc_dir
        self.img_dir = os.path.join(voc_dir, 'JPEGImages')
        self.label_dir = os.path.join(voc_dir, 'SegmentationClass')
        self.transform = transform

        txt_fname = os.path.join(voc_dir, 'ImageSets/Segmentation/', 'train.txt' if is_train else 'val.txt')
        with open(txt_fname, 'r') as f:
            self.images = f.read().split()
        
        self.img_fnames = [os.path.join(self.img_dir, f'{fname}.jpg') for fname in self.images]
        self.label_fnames = [os.path.join(self.label_dir, f'{fname}.png') for fname in self.images]

    def __len__(self, ):
        return len(self.images)

    def __getitem__(self, idx):
        img_fname = self.img_fnames[idx]
        label_fname = self.label_fnames[idx]
        img = Image.open(img_fname)
        label = Image.open(label_fname)

        if self.transform:
            img, label = self.transform((img, label))
        
        return img, label

def set_seed(seed: int=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
class BaseSegTransform:
    """
    定义用于分割任务的图像增强
    关键在于对 image 和 label 应用参数相同的 transform
    """
    support_transforms = [
        'RandomHorizontalFlip', 'RandomVerticalFlip'
    ]
    def __init__(self, transform, only_image=False):
        if not only_image:
            assert transform.__class__.__name__ in self.support_transforms, f'{transform.__class__.__name__} not supported'
        self.transform = transform
        self.only_image = only_image

    def __call__(self, sample_tuple):
        img, label = sample_tuple
        seed = torch.randint(0, 2**32-1, (1, )).item()
        set_seed(seed)
        img = self.transform(img)
        if not self.only_image:
            set_seed(seed)
            label = self.transform(label)
        return (img, label)

class SegRandomResizedCrop:
    def __init__(self, size, scale=(0.5, 1.0), ratio=(0.75, 1.33)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, sample_tuple):
        img, label = sample_tuple
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=self.scale, ratio=self.ratio)
        img = transforms.functional.resized_crop(img, i, j, h, w, self.size)
        label = transforms.functional.resized_crop(label, i, j, h, w, self.size, interpolation=Image.NEAREST)
        return img, label

class SegToTensor:
    def __call__(self, sample_tuple):
        img, label = sample_tuple
        img = transforms.functional.to_tensor(img)
        label = torch.as_tensor(np.array(label), dtype=torch.int64)
        return img, label

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = '/root/pytorch'
    voc_dir = os.path.join(ROOT_DIR, 'data/VOCdevkit/VOC2012/')

    train_transforms = [
        SegRandomResizedCrop((224, 224)),
        BaseSegTransform(transforms.RandomHorizontalFlip(), only_image=False),
        BaseSegTransform(transforms.RandomVerticalFlip(), only_image=False),
        BaseSegTransform(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), only_image=True),
        SegToTensor(),
        BaseSegTransform(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), only_image=True),
    ]

    train_transforms = transforms.Compose(train_transforms)
    dataset = MyVOCSegDataset(voc_dir, is_train=True, transform=train_transforms)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    for i in range(9):
        img, label = dataset[i]
        axes[i].imshow(transforms.ToPILImage()(img))
        axes[i].imshow(np.array(label), alpha=0.5)
        axes[i].axis('off')

#%%