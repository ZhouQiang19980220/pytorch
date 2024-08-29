#%%
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from loguru import logger
#%%
class MyDataset(Dataset):
    def __init__(self, len: int = 10, transform = None):
        self.len = len
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if index >= self.len:
            raise IndexError(f'Index {index} out of range')
        else:
            width = random.randint(100, 150)
            height = random.randint(100, 150)
            data = Image.fromarray(
                (np.random.rand(height, width, 3) * 255).astype(np.uint8)
            )
            label = self.get_label(height, width)
            if self.transform:
                data = self.transform(data)
            return data, label

    def get_label(self, height: int, width: int):
        return (height + width) % 3

def my_collate_fn(batch):
    data, label = list(zip(*batch))
    length = len(data)
    batch_size = (length, 3, 150, 150)
    batch_data = torch.zeros(batch_size)
    for i in range(length):
        h, w = data[i].shape[-2:]
        batch_data[i, :, :h, :w] = data[i]
    return batch_data, torch.tensor(label)

#%%
transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = MyDataset(len=10, transform=transform)
train_loader = DataLoader(
    train_dataset, batch_size=4, shuffle=True, collate_fn=my_collate_fn
)
for idx, (data, label) in enumerate(train_loader):
    if idx == 0:
        logger.debug(f'data: {data.shape}, label: {label.shape}')
        fig, axes = plt.subplots(2, 2)
        axes = axes.flatten()
        for i in range(4):
            axes[i].imshow(data[i].permute(1, 2, 0).numpy())
            axes[i].set_title(f'label: {label[i]}')
            axes[i].axis('off')

#%%