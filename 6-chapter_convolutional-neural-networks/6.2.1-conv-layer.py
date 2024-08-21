"""
自定义简化版卷积
1. 不考虑 stride
2. 不考虑 padding
3. 不考虑 dilation
4. 不考虑 bias
5. 不考虑 groups
6. 只考虑 2D 卷积
7. 只考虑正方形卷积核
"""

# %% import libraries
from loguru import logger
from typing import Optional
import torch
from torch import nn
# %% define a convolutional layer


class MyConv2d(nn.Module):
    """
    自定义简化版卷积
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._init_weight()

    def _init_weight(self, ):
        """
        初始化权重
        """
        kernel_shape = (
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        )
        self.weight = nn.Parameter(torch.empty(kernel_shape))

    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}'

    def _single_channel_conv(
        self,
        x: torch.Tensor,
        weight: torch,
        out: Optional[torch.Tensor] = None
    ):
        """
        单层卷积, 即输出特征图为单通道
        x.shape = (batch_size, in_channels, in_height, in_width)
        weight.shape = (1, in_channels, kernel_size, kernel_size)
        """
        batch_size, in_channels, in_height, in_width = x.shape
        is_return = out is None
        if out is None:
            out_height = in_height - self.kernel_size + 1
            out_width = in_width - self.kernel_size + 1
            out_shape = (batch_size, 1, out_height, out_width)
            out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        else:
            out_height, out_width = out.shape[2], out.shape[3]
        for i in range(out_height):
            for j in range(out_width):
                height_start = i
                height_end = i + self.kernel_size
                width_start = j
                width_end = j + self.kernel_size
                # x_slice.shape = (batch_size, in_channels, kernel_size, kernel_size)
                x_slice = x[:, :, height_start:height_end,
                            width_start:width_end]
                assert x_slice.shape == (
                    batch_size, in_channels, self.kernel_size, self.kernel_size)
                # 这里相乘会自动广播 weight 到 x_slice, 等价于对这个 batch 中的每个样本进行卷积
                # out[:, :, i, j].shape = (batch_size, 1)
                out[:, :, i, j] = torch.sum(
                    x_slice * weight, dim=(1, 2, 3)).unsqueeze(1)
        return out if is_return else None

    def _multi_channel_conv(self, x):
        """
        多层卷积, 即输出特征图为多通道
        x.shape = (batch_size, in_channels, in_height, in_width)
        weight.shape = (out_channels, in_channels, kernel_size, kernel_size)
        """
        batch_size, in_channels, in_height, in_width = x.shape
        out_shape = (
            batch_size,
            self.out_channels,
            in_height - self.kernel_size + 1,
            in_width - self.kernel_size + 1
        )
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        for c in range(self.out_channels):
            self._single_channel_conv(
                x, self.weight[c, :, :, :].unsqueeze(0), out[:, c, :, :].unsqueeze(1))
        return out

    def forward(self, x):
        return self._multi_channel_conv(x)


class MyPadding2d(nn.Module):
    """
    定义一个 简单的 padding 层
    仅考虑 0 padding
    """

    def __init__(self, padding: int):
        super().__init__()
        if padding < 0:
            raise ValueError("padding 必须是非负整数")
        self.padding = padding

    def forward(self, x):
        if self.padding == 0:
            return x
        batch_size, in_channels, in_height, in_width = x.shape
        out_height = in_height + 2 * self.padding
        out_width = in_width + 2 * self.padding
        out_shape = (batch_size, in_channels, out_height, out_width)
        out = torch.zeros(out_shape, device=x.device, dtype=x.dtype)

        height_start = self.padding
        height_end = in_height + self.padding
        width_start = self.padding
        width_end = in_width + self.padding

        out[:, :, height_start:height_end, width_start:width_end] = x
        return out

    def extra_repr(self):
        return f'padding={self.padding}'


class MyConvPadding2d(MyConv2d):
    """
    定义一个考虑 padding 的卷积层
    仅考虑 0 padding
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__(in_channels, out_channels, kernel_size)
        self.padding_layer = MyPadding2d(padding)

    def forward(self, x):
        x = self.padding_layer(x)
        return super().forward(x)

    def extra_repr(self):
        return f'{super().extra_repr()}, padding={self.padding_layer.padding}'


if __name__ == '__main__':
    import unittest

    class TestMyConv2d(unittest.TestCase):
        def test_single_channel_conv(self):
            myConv2d = MyConv2d(3, 1, 3)
            conv2d = nn.Conv2d(3, 1, 3, bias=False)
            # (out_channels, in_channels, kernel_size, kernel_size)
            logger.debug(conv2d.weight.shape)
            logger.debug(myConv2d.weight.shape)

            for i in range(100):    # 100 次测试
                # 初始化权重
                nn.init.uniform_(conv2d.weight)
                myConv2d.weight.data = conv2d.weight.data

                # 构造输入
                # (batch_size, in_channels, in_height, in_width)
                x = torch.randn(1, 3, 5, 5)
                y_conv2d = conv2d(x)
                y_myConv2d = myConv2d._single_channel_conv(x, myConv2d.weight)
                self.assertTrue(torch.allclose(
                    y_conv2d, y_myConv2d, atol=1e-3))

        def test_forward(self, ):
            conv2d = nn.Conv2d(3, 4, 3, bias=False)
            myConv2d = MyConv2d(3, 4, 3)

            nn.init.uniform_(conv2d.weight)
            myConv2d.weight.data = conv2d.weight.data

            x = torch.randn(100, 3, 3, 3)
            y_conv2d = conv2d(x)
            y_myConv2d = myConv2d(x)
            logger.debug(f'{y_myConv2d.shape=}')
            self.assertTrue(torch.allclose(y_conv2d, y_myConv2d, atol=1e-3))

    unittest.main()
