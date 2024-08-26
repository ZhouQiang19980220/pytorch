"""
自定义U-Net模型
"""
#%% import libraries
import torch
from torch import nn

from loguru import logger

#%% 定义ConvBlock
class ConvBlock(nn.Module):
    """
    conv -> bn -> relu -> conv -> bn -> relu
    kernel size = 3, padding = 1
    不改变输入的尺寸
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class DownSampleBlock(nn.Module):
    """
    conv -> bn -> relu -> conv -> bn -> relu -> maxpool
    两倍下采样
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # TODO: 为什么这里要返回两个值
        x = self.conv(x)
        p = self.pool(x)
        return x, p

class UpSampleBlock(nn.Module):
    """
    upsample -> concat -> conv -> bn -> relu -> conv -> bn -> relu
    upsample 使用转置卷积
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        # TODO: 这里的 skip 是什么
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    """
    U-Net 输出尺寸和输入尺寸相同, 通道数由 out_channels 指定
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down1 = DownSampleBlock(in_channels, 64)
        self.down2 = DownSampleBlock(64, 128)
        self.down3 = DownSampleBlock(128, 256)
        self.down4 = DownSampleBlock(256, 512)
        self.bottleneck = ConvBlock(512, 1024)
        self.up1 = UpSampleBlock(1024, 512)
        self.up2 = UpSampleBlock(512, 256)
        self.up3 = UpSampleBlock(256, 128)
        self.up4 = UpSampleBlock(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1, p1 = self.down1(x)
        d2, p2 = self.down2(p1)
        d3, p3 = self.down3(p2)
        d4, p4 = self.down4(p3)
        bottleneck = self.bottleneck(p4)
        u1 = self.up1(bottleneck, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        return self.final_conv(u4)

if __name__ == "__main__":
    model = UNet(3, 1)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    logger.info(y.shape)
