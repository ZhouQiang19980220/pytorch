#%%
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from loguru import logger

import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]   # get the image name
landmarks = landmarks_frame.iloc[n, 1:] # get the landmarks
landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)        # convert to numpy array

logger.info(f'Image name: {img_name}')  # person-7.jpg
logger.info(f'Landmarks shape: {landmarks.shape}')  # (68, 2)
logger.info(f'First 4 Landmarks: \n {landmarks[:4]}')    # four landmarks

def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()
img = io.imread(os.path.join('data/faces/', img_name))  # read the image as numpy array
show_landmarks(img, landmarks)

#%% define a custom dataset
class FaceLandmarksDataset(Dataset):

    def __init__(
        self, 
        csv_file,
        root_dir,
        transform=None
    ):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):    # 将torch.tensor转换为list
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])   
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/')

fig = plt.figure()
for i, sample in enumerate(face_dataset):
    logger.info(
        f'{i}: {sample["image"].shape}, {sample["landmarks"].shape}')
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()  # This function is used to automatically adjust the spacing between subplots in a figure to optimize the layout.
    ax.set_title(f'Sample #{i}')
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break


#%% custom transforms
class Rescale(object):  # 自定义某种 transforms, 必须实现 __call__ 方法
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}