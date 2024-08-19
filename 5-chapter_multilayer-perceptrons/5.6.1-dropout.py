#%% import libraries
import unittest
import torch
from torch import nn
from d2l import torch as d2l

nn.Dropout()
#%% 实现自己的Dropout
class Dropout(nn.Module):   # 这里需要继承nn.Module

    def __init__(self, dropout_p):
        super().__init__()
        assert 0 <= dropout_p <= 1, 'dropout probability has to be between 0 and 1'
        self.dropout_p = dropout_p

    def forward(self, X):
        if self.training:   # 训练模式, 以概率self.dropout_p丢弃X中的元素
            if self.dropout_p == 0:
                return X
            elif self.dropout_p == 1:   # dropout_p=1, 表示全部丢弃。这里直接返回全0, 避免后续的除以 0
                return torch.zeros_like(X)
            mask = torch.rand(X.shape) > self.dropout_p
            # 这里使用mask做element-wise乘法，以实现丢弃元素, 对 GPU 计算更友好 
            return X * mask / (1.0 - self.dropout_p)
        else:   # 非训练模式，直接返回X
            return X

class TestDropout(unittest.TestCase):

    def test_forward(self):
        dropout = Dropout(0.5)
        # 训练模式
        dropout.train()
        print(f'{dropout.training=}')
        X = torch.randn(size=(100, 100))
        Y = dropout(X)
        self.assertEqual(X.shape, Y.shape)  # 确保形状不变
        # 统计 Y 中值为0的元素的比例，接近0.5
        dropout_p = torch.isclose(Y, torch.zeros_like(Y)).float().mean()
        print(f'{dropout_p=}')
        # 设置容差为0.1
        self.assertTrue(torch.isclose(torch.tensor(0.5), dropout_p, atol=0.1))

        # 设置dropout_p=0, 不丢弃元素
        dropout = Dropout(0)
        Y = dropout(X)
        self.assertTrue(torch.allclose(X, Y))

        # 设置dropout_p=1, 全部丢弃
        dropout = Dropout(1)
        Y = dropout(X)
        self.assertTrue(torch.allclose(torch.zeros_like(Y), Y))
        
        # 非训练模式
        dropout.eval()
        print(f'{dropout.training=}')
        Y = dropout(X)
        self.assertTrue(torch.allclose(X, Y))


class MLPDropout(d2l.Classifier):

    def __init__(
        self, 
        num_inputs: int,
        num_outputs: int,
        num_hiddens: int, 
        dropout: float, 
        lr: float = 0.1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hiddens, num_outputs)
        )
        self.lr = lr

    def forward(
        self, 
        X: torch.Tensor):
        return self.layers(X)

def main():
    # 定义模型参数
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    dropout = 0.5
    net = MLPDropout(num_inputs, num_outputs, num_hiddens, dropout)
    print(f'{net=}')

    # 准备数据
    batch_size, lr, num_epochs = 256, 0.1, 10
    data = d2l.FashionMNIST(batch_size)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    trainer = d2l.Trainer(
        max_epochs=num_epochs,
        num_gpus=1
    )
    trainer.fit(net, data)

    d2l.plt.show()


if __name__ == '__main__':
    # unittest.main()
    main()


#%%