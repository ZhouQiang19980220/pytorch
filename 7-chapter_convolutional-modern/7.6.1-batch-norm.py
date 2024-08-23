"""
自定义 BatchNorm 层
"""

#%% import libraries
from loguru import logger
import torch
from torch import nn

#%%
class MyBatchNorm1d(nn.Module):
    """
    自定义 BatchNorm1d 层
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(1, num_features, ))
        self.beta = nn.Parameter(torch.zeros(1, num_features, ))

        self.running_mean = torch.zeros(1, num_features)
        self.running_var = torch.ones(1, num_features)

    def forward(self, x):
        # 检查输入合法性
        assert len(x.shape) == 2, f'BatchNorm1d only support 2D input, but get {len(x.shape)}D input'

        self.running_mean = self.running_mean.to(x.device, x.dtype)
        self.running_var = self.running_var.to(x.device, x.dtype)

        # 训练模式
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            # 方差计算的两个注意点
            # 1. 使用有偏估计, 对齐 PyTorch 的实现
            # 2. 加上一个小的 eps, 防止除以 0
            var = x.var(dim=0, unbiased=False, keepdim=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            x_hat = (x - mean) / (var.sqrt() + self.eps)
        else:
            x_hat = (x - self.running_mean) / self.running_var.sqrt()
        return self.gamma * x_hat + self.beta

    def extra_repr(self):
        return f'num_features={self.gamma.shape[1]}, eps={self.eps}, momentum={self.momentum}'


class MyBatchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        self.running_mean = torch.zeros(1, num_features, 1, 1)
        self.running_var = torch.ones(1, num_features, 1, 1)

    def forward(self, x):
        assert len(x.shape) == 4, f'BatchNorm2d only support 4D input, but get {len(x.shape)}D input'

        self.running_mean = self.running_mean.to(x.device, x.dtype)
        self.running_var = self.running_var.to(x.device, x.dtype)
        
        if self.training:   # 训练模式
            mean = x.mean(dim=(0, 2, 3), keepdim=True)    # 在通道维度上求均值
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            # 这里下标写None, 是为了让 mean 和 var 的维度和 x 一致。
            x_hat = (x - mean) / (var.sqrt() + self.eps)
        else:   # 测试模式
            x_hat = (x - self.running_mean) / self.running_var.sqrt()
        return self.gamma * x_hat + self.beta

    def extra_repr(self):
        return f'num_features={self.gamma.shape[1]}, eps={self.eps}, momentum={self.momentum}'

#%% 测试
x = torch.randn(3, 4)
myBatchNorm1d = MyBatchNorm1d(4)
myBatchNorm1d.train()
y_my = myBatchNorm1d(x)
logger.info(f'{y_my.shape=}')
# 验证输出的均值和方差
logger.info(f'{y_my.mean(dim=0)=}')
logger.info(f'{y_my.var(dim=0, unbiased=False)=}')

batchNorm1d = nn.BatchNorm1d(4)
batchNorm1d.train()
y = batchNorm1d(x)
logger.info(f'{y.shape=}')
# 验证输出的均值和方差
logger.info(f'{y.mean(dim=0)=}')
logger.info(f'{y.var(dim=0, unbiased=False)=}')

x = torch.randn(2, 3, 4, 4)
myBatchNorm2d = MyBatchNorm2d(3)
myBatchNorm2d.train()
y_my = myBatchNorm2d(x)
logger.info(f'{y_my.shape=}')
# 验证输出的均值和方差
logger.info(f'{y_my.mean(dim=(0, 2, 3))=}')
logger.info(f'{y_my.var(dim=(0, 2, 3), unbiased=False)=}')

batchNorm2d = nn.BatchNorm2d(3)
batchNorm2d.train()
y = batchNorm2d(x)
logger.info(f'{y.shape=}')
# 验证输出的均值和方差
logger.info(f'{y.mean(dim=(0, 2, 3))=}')
logger.info(f'{y.var(dim=(0, 2, 3), unbiased=False)=}')
#%%