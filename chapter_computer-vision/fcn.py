"""
动手实现FCN: Fully Convolutional Networks
"""

#%% import libraries
import functools
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import lightning as pl
from d2l import torch as d2l
from loguru import logger

from vocdataset import MyVOCSegDataset, SegTransform
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
    validate_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    validate_trans = SegTransform(validate_trans, only_image=False)
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

def get_lit_fcn():
    backbone = torchvision.models.resnet18(pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    hidden_dim = backbone[-1][-1].bn2.num_features
    fcn = FCN(backbone, hidden_dim, num_classes=21)
    lit_fcn = LitFCN(fcn)
    return lit_fcn

#%% test the FCN model
torch.set_float32_matmul_precision('medium')
# 获取数据加载器
dataloader = get_voc_dataloader(num_workers=127)
# 获取模型
lit_fcn = get_lit_fcn()
# 训练模型
trainer = pl.Trainer(max_epochs=1, log_every_n_steps=1)
trainer.fit(lit_fcn, dataloader['train'], dataloader['val'])

#%%
# 预测部分
from matplotlib import pyplot as plt
dataloader = get_voc_dataloader(num_workers=127)
img, mask = next(iter(dataloader['val']))
pred = lit_fcn.nn_fcn(img)
pred = pred.argmax(1)
img = img.permute(0, 2, 3, 1)
# mask = mask.permute(0, 2, 3, 1)
# pred = pred.permute(0, 2, 3, 1)
img, mask, pred = img[0], mask[0], pred[0]
img = img.numpy()
mask = mask.numpy()
pred = pred.numpy()

#%%
# 绘制图像
import matplotlib.colors as mcolors
plt.subplot(131)
plt.imshow(img, cmap='gray', norm=mcolors.Normalize(vmin=img.min(), vmax=img.max()))
plt.subplot(132)
plt.imshow(mask, cmap='gray', norm=mcolors.Normalize(vmin=mask.min(), vmax=mask.max()))
plt.subplot(133)
plt.imshow(pred, cmap='gray', norm=mcolors.Normalize(vmin=pred.min(), vmax=pred.max()))
plt.show()

# %%