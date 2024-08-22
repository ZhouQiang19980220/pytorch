"""
自定义 NiN 
NiN 最大的创新是串联多个由卷秮层和 1x1 卷积层构成的 NiN 块
1x1 卷积层充当通道之间的全连接层
"""
#%% import libraries
import torch
from torch import nn
from d2l import torch as d2l

#%% NiN块
class NiNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(NiNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            # 后面两个 1x1 卷积层当做全连接层使用
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)

#%% NiN模型
class NiN(d2l.Classifier): 
    """
    NiN 模型
    """
    def __init__(self, in_features=1, out_features=10, lr=0.1):
        super().__init__()
        self.lr = lr
        # input: 1x224x224
        self.net = nn.Sequential(
            # 1x224x224 -> 96x54x54
            NiNBlock(in_features, 96, kernel_size=11, stride=4, padding=0),
            # 96x54x54 -> 96x26x26
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 96x26x26 -> 256x26x26
            NiNBlock(96, 256, kernel_size=5, stride=1, padding=2),
            # 256x26x26 -> 256x12x12
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 256x12x12 -> 384x12x12
            NiNBlock(256, 384, kernel_size=3, stride=1, padding=1),
            # 384x12x12 -> 384x5x5
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 384x5x5 -> 10x5x5
            NiNBlock(384, out_features,  kernel_size=3, stride=1, padding=1),
            # 10x5x5 -> 10x1x1
            nn.AdaptiveAvgPool2d((1, 1)),
            # 10x1x1 -> 10  这里的 10 是输出的类别数
            nn.Flatten()    
        )

    def forward(self, x):
        return self.net(x)

#%% test
nin = NiN()
model = NiN(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)

#%%