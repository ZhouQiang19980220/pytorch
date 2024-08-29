#%%
import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from loguru import logger

from vocdataset import MyVOCSegDataset
from fcn import get_nn_fcn

#%% setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # __file__表示当前文件
PARENT_DIR = os.path.dirname(BASE_DIR)
logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"PARENT_DIR: {PARENT_DIR}")

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
device = get_device()
logger.info(f"device: {device}")

plt.ion()   # interactive mode
logger.info("Interactive mode enabled")

#%% load and preprocess image
# img_fname = 'img/cat1.jpg'
# img_fname = 'img/cat2.jpg'
img_fname = 'img/cat3.jpg'
# img_fname = 'img/catdog.jpg'
img = Image.open(os.path.join(PARENT_DIR, img_fname))
logger.info(f"img.size: {img.size}")

preprocess = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])
# 
img_tensor = preprocess(img).unsqueeze(0).to(device)
# %% load and set model
# fcn = torchvision.models.segmentation.fcn_resnet50(weights=torchvision.models.segmentation.FCN_ResNet50_Weights)
fcn = get_nn_fcn()
fcn.load_state_dict(torch.load('/root/pytorch/chapter_computer-vision/fcn.pth'))
fcn = fcn.to(device)
fcn.eval()
logger.info(f"fcn: \n{fcn}")

# %% inference
with torch.no_grad():
    out_4d= fcn(img_tensor) # (B, C, H, W)
logger.info(f"out.size: {out_4d.size()}")
#%%
out = out_4d.squeeze(0)
logger.info(f"out.size: {out.size()}")
prediction = out.argmax(0)
# %% display
def imshow(imgs, titles=None):
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    rows, cols = len(imgs), 1
    if titles is None:
        titles = [f"img{i}" for i in range(len(imgs))]
    assert len(imgs) <= len(titles)
    _, axes = plt.subplots(rows, cols, figsize=(rows*5, cols*5))
    axes.flatten()
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

prediction_np = prediction.cpu().numpy()
imshow([img, prediction_np], titles=['img', 'prediction'])

# %%
