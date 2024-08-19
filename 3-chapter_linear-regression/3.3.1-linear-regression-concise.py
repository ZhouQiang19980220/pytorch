"""
线性回归的简洁实现
"""

# %% import library
from typing import Tuple
import torch
from torch import nn
from torch.utils import data
from loguru import logger
from d2l import torch as d2l
# 定义超参数
lr = 0.03
num_epochs = 3
batch_size = 16

# 生成数据集
num_examples = 1000
true_w = torch.tensor([2, -3.4])
true_b = torch.tensor(4.2)
features, labels = d2l.synthetic_data(true_w, true_b, num_examples)
# %% 调用已有的框架来读取数据


def load_array(
        data_arrays: Tuple[torch.Tensor],
        batch_size: int,
        is_train: bool = True):
    """
    pass
    """
    dataset = data.TensorDataset(*data_arrays)
    # shuffle=True 表示打乱数据; 训练集需要打乱数据, 测试集不需要
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# %% 定义神经网络
net = nn.Sequential(nn.Linear(2, 1))

# 初始化神经网络的参数
# net[0]表示线性层
# 线性层的参数有权重和偏差
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# %% 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

# %% 开始训练
data_iter = load_array((features, labels), batch_size, is_train=True)
logger.info('starting training ...')
for epoch in range(num_epochs):
    for batch_data in data_iter:
        batch_x, batch_y = batch_data
        y_hat = net(batch_x)    # 前向传播
        loss_value = loss_fn(y_hat, batch_y)    # 计算损失
        optimizer.zero_grad()   # 梯度清零
        loss_value.backward()   # 反向传播
        optimizer.step()        # 参数更新

    logger.info(f"{epoch=}, loss={loss_value:.4f}")
logger.info('finished training.')
logger.info(f'{true_w=}, {true_b=}')
logger.info(f'w={net[0].weight.data}, b={net[0].bias.data}')

# %%
