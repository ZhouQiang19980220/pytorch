"""
实现LeNet-5模型
"""

# %% import libraries
from loguru import logger
import torch
from torch import nn

# %% define a LeNet-5 model


class LeNet(nn.Module):

    def __init__(self, ):
        super().__init__()
        # input shape: (batch_size, 1, 28, 28)

        self.conv = nn.Sequential(
            # (batch_size, 6, 28, 28)
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            # (batch_size, 6, 14, 14)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (batch_size, 16, 10, 10)
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            # (batch_size, 16, 5, 5)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            # (batch_size, 120)
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            # (batch_size, 84)
            nn.Linear(120, 84),
            nn.ReLU(),
            # (batch_size, 10)
            nn.Linear(84, 10)
        )

    def forward(self, x: torch.Tensor):
        # expected input shape: (batch_size, 1, 28, 28)
        return self.fc(self.flatten(self.conv(x)))


if __name__ == '__main__':
    lenet = LeNet()
    logger.info(f'lenet = \n{lenet}')
    # 打印每一层的输入和输出形状
    x = torch.empty(1, 1, 28, 28)
    logger.info(f'input shape: {x.shape}')
    for layer in lenet.conv:
        x = layer(x)
        logger.info(f'{layer.__class__.__name__}, output shape: {x.shape}')

    x = lenet.flatten(x)
    logger.info(f'Flatten, output shape: {x.shape}')

    for layer in lenet.fc:
        x = layer(x)
        logger.info(f'{layer.__class__.__name__}, output shape: {x.shape}')

    for module in lenet.modules():
        logger.info(f'{module.__class__.__name__}, {module.extra_repr()}')
