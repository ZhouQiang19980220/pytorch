"""
A tutorial of the torchvision.transforms module
"""

#%% import libraries
import torch
from torchvision import transforms
from PIL import Image
from loguru import logger

#%% read an image
img_fname = '/root/pytorch/img/cat3.jpg'
img = Image.open(img_fname)
logger.info(f"Image size: {img.size}")
img.show()

#%% define a transform
# image classification
img_transform = transforms.RandomHorizontalFlip(p=1)(img)
img_transform.show(title='RandomHorizontalFlip')

img_transform = transforms.RandomVerticalFlip(p=1)(img)
img_transform.show(title='RandomVerticalFlip')

img_transform = transforms.RandomRotation(45)(img)
img_transform.show(title='RandomRotation')

img_transform = transforms.RandomResizedCrop(224)(img)
img_transform.show(title='RandomResizedCrop')
#%% detection
from torchvision import tv_tensors

boxes = torch.tensor([100, 100, 200, 200])
boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=img.size)
img.add_boxes(boxes)
img.show()

#%%
