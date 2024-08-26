"""
自定义简化版的转置卷积层
1. 不考虑步长
2. 不考虑填充
3. 不考虑多通道
"""
#%% import libraries
import unittest
import torch
from torch import nn
from loguru import logger
#%% function: transposed_conv
def transposed_conv(X: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    X: input tensor, shape (h, w)
    kernel: kernel tensor, shape (k_h, k_w)
    """
    k_h, k_w = kernel.shape
    h, w = X.shape
    Y = torch.zeros((h + k_h - 1, w + k_w - 1)) # 使用全零填充来占位
    for i in range(h):
        for j in range(w):
            Y[i: i + k_h, j: j + k_w] += X[i, j] * kernel
    return Y

#%% test function: transposed_conv
class TestTransposedConv(unittest.TestCase):
    def test_transposed_conv(self):
        X = torch.tensor([
            [0.0, 1.0], 
            [2.0, 3.0]])
        K = torch.tensor([
            [0.0, 1.0], 
            [2.0, 3.0]])
        Y = transposed_conv(X, K)
        logger.debug(f'Y: \n{Y}')
        self.assertTrue(torch.allclose(Y, torch.tensor([
            [0.0, 0.0, 1.0], 
            [0.0, 4.0, 6.0], 
            [4.0, 12.0, 9.0]])))

        # 调用PyTorch的转置卷积层
        X = X[None, None, :, :]
        K = K[None, None, :, :]
        # pytorch的转置卷积层
        convtrans = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
        convtrans.weight.data = K
        Y = convtrans(X)
        logger.debug(f'Y: \n{Y}')
        self.assertTrue(torch.allclose(Y, torch.tensor([
            [0.0, 0.0, 1.0], 
            [0.0, 4.0, 6.0], 
            [4.0, 12.0, 9.0]]).reshape(1, 1, 3, 3)))

# 测试 pytorch 的转置卷积层
class TestConvTranspose2d(unittest.TestCase):

    def test_conv_transpose2d_padding(self, ):
        """
        测试 padding 参数
        """
        # 这里设置 padding=1，表示把输出特征图的最外面一圈删除
        # 所以增加 padding=1，输出特征图的尺寸减少
        convtrans = nn.ConvTranspose2d(1, 1, kernel_size=3, padding=1, bias=False)
        x = torch.rand(size=(1, 1, 3, 3))
        y = convtrans(x)
        logger.debug(f'y.shape: \n{y.shape}')
        self.assertEqual(y.shape, (1, 1, 3, 3))

    def test_conv_transpose2d_stride(self, ):
        """
        测试 stride 参数
        增加 stride=2, 输出特征图的尺寸会增加
        """
        convtrans = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, bias=False)
        x = torch.rand(size=(1, 1, 3, 3))
        y = convtrans(x)
        logger.debug(f'y.shape: \n{y.shape}')
        self.assertEqual(y.shape, (1, 1, 7, 7))

    def test_conv_transpose2d_conv(self, ):
        """
        假设 stride=1，其余参数 conv 和 convtrans 一致
        那么 convtrans(conv(x)) 的输出特征图尺寸应该和 x 一致
        但是里面的值全部变了, 只是在形状上可逆
        """
        x_shape = (1, 1, 16, 16)
        x = torch.rand(size=x_shape)
        conv = nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        convtrans = nn.ConvTranspose2d(
            1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.assertEqual(convtrans(conv(x)).shape, x_shape)

if __name__ == "__main__":
    unittest.main()