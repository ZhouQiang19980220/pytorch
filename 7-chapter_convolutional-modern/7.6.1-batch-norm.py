"""
自定义 BatchNorm 层
"""

#%% import libraries
import torch
from torch import nn

#%% 自定义 BatchNorm 层
class BatchNorm2d(nn.Module):
    """
    自定义简化版 BatchNorm2d 层
    """

    def __init__(
        self, 
        num_features,   # 输入特证数量
        eps=1e-5,       # 防止分母为 0
        momentum=0.1    # 用于计算移动平均的均值和方差
        ):
        super().__init__()

        # 初始化参数
        self.eps = eps
        self.momentum = momentum
        