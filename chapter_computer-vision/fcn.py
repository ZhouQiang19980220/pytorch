"""
动手实现FCN: Fully Convolutional Networks
"""

#%% import libraries
import os
import functools
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import lightning as pl
from d2l import torch as d2l
from loguru import logger

from vocdataset import BaseSegTransform, SegRandomResizedCrop, SegToTensor
from vocdataset import MyVOCSegDataset

from matplotlib import pyplot as plt
from tqdm import tqdm
#%% 定义双线性插值层

#%% define the FCN model
class FCN(nn.Module):
    def __init__(self, backbone, hidden_dim, num_classes=10):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.conv = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        self.head = functools.partial(F.interpolate, mode='bilinear')

    def forward(self, x):
        h, w = x.shape[2:]
        x = self.backbone(x)
        x = self.conv(x)
        x = self.head(x, size=(h, w))
        return x

class LitFCN(pl.LightningModule):
    
    def __init__(self, nn_fcn: nn.Module):
        super().__init__()
        self.nn_fcn = nn_fcn

    def loss_fn(self, pred, mask):
        return F.cross_entropy(pred, mask, reduction='none').mean(1).mean(1).mean()

    def training_step(self, batch, batch_idx):
        img, mask = batch
        pred = self.nn_fcn(img)
        loss = self.loss_fn(pred, mask)
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(pred, mask))
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        pred = self.nn_fcn(img)
        loss = self.loss_fn(pred, mask)
        self.log('val_loss', loss)
        self.log('train_acc', self.accuracy(pred, mask))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.nn_fcn.parameters(), lr=1e-3)

    def accuracy(self, pred, mask):
        return (pred.argmax(1) == mask).float().mean()

def get_voc_dataloader(
    voc_dir: str = '../data/VOCdevkit/VOC2012', 
    batch_size=32, 
    num_workers = 4):
    training_trans = [SegRandomResizedCrop((224, 224))]
    training_trans.extend([
        BaseSegTransform(transforms.RandomHorizontalFlip()),
        BaseSegTransform(transforms.RandomVerticalFlip()),
    ])
    training_trans.extend([
        BaseSegTransform(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.01), only_image=True),
        SegToTensor(), 
        BaseSegTransform(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), only_image=True)
    ])

    training_trans = transforms.Compose(training_trans)
    validate_trans = transforms.Compose([
        SegRandomResizedCrop((224, 224)),
        SegToTensor(),
        BaseSegTransform(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), only_image=True)
    ])
    train_dataset = MyVOCSegDataset(voc_dir, is_train=True, transform=training_trans)
    logger.debug('train_dataset loaded, length: %d', len(train_dataset))
    val_dataset = MyVOCSegDataset(voc_dir, is_train=False, transform=validate_trans)
    logger.debug('val_dataset loaded, length: %d', len(val_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    dataloader = {
        'train': train_loader,
        'val': val_loader
    }
    return dataloader

def get_nn_fcn():
    # 使用预训练的 resnet18 作为 backbone
    # backbone = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone = torchvision.models.resnet18(pretrained=True)
    # 去掉最后两层: avgpool 和 fc
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    # 获取最后一层的输出通道数
    hidden_dim = backbone[-1][-1].bn2.num_features
    # 定义 FCN 模型
    fcn = FCN(backbone, hidden_dim, num_classes=21)
    return fcn

def get_lit_fcn():
    fcn = get_nn_fcn()
    lit_fcn = LitFCN(fcn)
    return lit_fcn

def check_label(label: torch.Tensor):
    numel = label.numel()
    for i in range(21):
        num = (label == i).sum().item()
        numel -= num
    numel -= (label == 255).sum().item()
    if numel != 0:
        logger.warning(f'numel: {numel}, expected 0')


#%% test the FCN model
if __name__ == '__main__':
    ROOT = '/root/pytorch'
    voc_dir = os.path.join(ROOT, 'data/VOCdevkit/VOC2012')
    # dataloaders = get_voc_dataloader(voc_dir, batch_size=4, num_workers=0)
    # plt.ion()
    # fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    # axes = axes.flatten()
    # for batch in dataloaders['train']:
    #     img, label = batch
    #     logger.debug(f'{img.shape=}, {label.shape=}')
    #     logger.debug(f'{img.dtype=}, {label.dtype=}')
    #     for i in range(4):
    #         axes[i].imshow(img[i].permute(1, 2, 0))
    #         axes[i].imshow(label[i], alpha=0.5)
    #         axes[i].set_title(f'{label[i].shape=}')
    #         check_label(label[i])
    #     plt.tight_layout()
    #     break

    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug(f'{device=}')
    dataloader = get_voc_dataloader(voc_dir, batch_size=16, num_workers=128)
    nn_fc = get_nn_fcn()
    nn_fc.to(device)
    optimizer = torch.optim.Adam(nn_fc.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    best_loss = float('inf')
    for epoch in range(num_epochs):
        training_loss = 0.0
        training_samples = 0
        nn_fc.train()
        for batch in tqdm(dataloader['train'], desc=f'training at {epoch}', leave=False):
            img, label = batch
            img, label = img.to(device), label.to(device)
            output = nn_fc(img)
            loss_value = loss_fn(output, label)
            training_loss += loss_value.item() * img.shape[0]
            training_samples += img.shape[0]
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        training_loss /= training_samples

        val_loss = 0.0
        val_samples = 0
        nn_fc.eval()
        for batch in tqdm(dataloader['val'], desc=f'validating at {epoch}', leave=False):
            img, label = batch
            img, label = img.to(device), label.to(device)
            output = nn_fc(img)
            loss_value = loss_fn(output, label)
            val_loss += loss_value.item() * img.shape[0]
            val_samples += img.shape[0]
        val_loss /= val_samples
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(nn_fc.state_dict(), 'fcn.pth')
            logger.info(f'save model to fcn.pth at epoch {epoch}')
        logger.debug(f'[epoch={epoch:3d}], training_loss={training_loss:.4f}, val_loss={val_loss:.4f}')
# %%