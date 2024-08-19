#%% import libraries
import torch
import torch.nn as nn

#%% 单隐藏层感知机从 0 实现
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
        h = torch.relu(
            torch.add(torch.matmul(x, self.W1), self.b1)`` 
        )
        o = torch.add(torch.matmul(h, self.W2), self.b2)
        return o

# %%
if __name__ == "__main__":
    mlp = MLP(2, 3, 4)
    for p in mlp.params:
        print(f'{p.shape=}:')

#%%
pass