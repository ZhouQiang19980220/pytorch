from typing import Union, Sequence
from torch import nn

from utils import init_weights


class BaseHousePriceNet(nn.Module):
    """
    所有房价预测网络的基类, 抽象类, 不能被实例化
    """

    def __init__(self):
        super().__init__()

    def init_net(self):
        """
        初始化网络
        """
        raise NotImplementedError('init_net not implemented')

    def forward(self, x):
        """
        前向传播
        """
        return self.net(x)


class HousePriceMLP(BaseHousePriceNet):
    """
    使用多层感知机的房价预测网络
    """

    def __init__(
        self,
        input_dim: int = 330,
        hidden_dims: Union[Sequence, int] = (256, 128),
        output_dim: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.net = self.init_net()
        self.net.apply(init_weights)

    def init_net(self):
        layers = []
        for i, hidden_dim in enumerate(self.hidden_dims):
            if i == 0:
                layers.append(nn.Linear(self.input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(self.hidden_dims[i-1], hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
        return nn.Sequential(*layers)
