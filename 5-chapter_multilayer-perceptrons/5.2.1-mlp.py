# %% import libraries
import torch
import torch.nn as nn
from d2l import torch as d2l
# %% 单隐藏层感知机从 0 实现


class MLP():

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.init_param()

    def init_param(self, ) -> None:
        self.W1 = nn.Parameter(
            torch.randn(self.input_dim, self.hidden_dim)
        )
        self.b1 = nn.Parameter(
            torch.randn(self.hidden_dim)
        )
        self.W2 = nn.Parameter(
            torch.randn(self.hidden_dim, self.output_dim)
        )
        self.b2 = nn.Parameter(
            torch.randn(self.output_dim)
        )
        self.params = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.input_dim)
        # h.shape = (batch_size, hidden_dim)
        h = torch.relu(
            torch.add(torch.matmul(x, self.W1), self.b1)
        )
        # o.shape = (batch_size, output_dim)
        o = torch.add(torch.matmul(h, self.W2), self.b2)
        return o
