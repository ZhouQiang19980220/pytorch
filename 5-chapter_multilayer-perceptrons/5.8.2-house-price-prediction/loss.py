import torch
from torch import nn


class BaseHousePriceLoss(nn.Module):
    """
    所有的房价预测损失函数的基类
    """

    def __init__(
        self,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.reduction = reduction

    def forward(self, outputs, labels):
        """
        计算损失
        """
        loss = self.loss(outputs, labels)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def loss(self, outputs, labels):
        """
        实际的损失计算
        """
        raise NotImplementedError('loss function not implemented')  # 抽象方法


# 定义一个相对误差的损失函数
class RelativeHousePriceLoss(BaseHousePriceLoss):
    """
    相对误差损失
    """

    def __init__(
        self,
        reduction: str = 'mean'
    ):
        super().__init__(reduction)

    def loss(self, outputs, labels):
        # 这里对 outputs 进行简单的处理，使得 outputs 和 labels 的形状和数据类型一致
        return torch.abs(
            outputs.reshape(labels.shape).type(labels.dtype) - labels
        ) / labels
