"""
自定义一个简化的池化层
1. 仅支持最大池化
2. 仅支持 stride = kernel_size
3. 仅支持正方形池化核
4. 仅支持 2D 池化
"""
# %% import libraries
from loguru import logger
import torch
from torch import nn

# %% define a pooling layer


class MyPool2d(nn.Module):
    """
    自定义简化版池化
    """

    def __init__(
        self,
        kernel_size: int = 2
    ):
        super().__init__()
        self.kernel_size = kernel_size

    def extra_repr(self):
        return f'kernel_size={self.kernel_size}'

    def forward(self, x: torch.Tensor):
        """
        前向传播
        """
        batch_size, in_channels, in_height, in_width = x.shape
        out_height = in_height // self.kernel_size
        out_width = in_width // self.kernel_size
        out_shape = (batch_size, in_channels, out_height, out_width)
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        for i in range(out_height):
            for j in range(out_width):
                height_start = i * self.kernel_size
                height_end = height_start + self.kernel_size
                width_start = j * self.kernel_size
                width_end = width_start + self.kernel_size
                x_slice = x[:, :, height_start:height_end,
                            width_start:width_end]

                # 这里使用value是因为max返回的是一个元组，第一个元素是最大值，第二个元素是最大值的索引
                # 在宽度方向上求最大值
                max_pool = torch.max(x_slice, dim=3).values
                # 在高度方向上求最大值
                out[:, :, i, j] = torch.max(max_pool, dim=2).values
        return out


if __name__ == '__main__':
    import unittest

    class TestMyPool2d(unittest.TestCase):

        def test_forward(self):
            for i in range(100):
                x = torch.randn(2, 3, 4, 4)
                myPool2d = MyPool2d()
                out_myPool2d = myPool2d(x)

                # 缺省情况下，stride = kernel_size
                pool2d = nn.MaxPool2d(kernel_size=2)
                out_pool2d = pool2d(x)

                assert torch.allclose(out_myPool2d, out_pool2d)

    unittest.main()
