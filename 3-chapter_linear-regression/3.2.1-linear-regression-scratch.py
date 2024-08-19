"""
从 0 实现线性回归
"""
# %% import library
# 1. standard library
# 2. third-party library
# 3. local library
import random
from typing import Union, Tuple, Generator, Callable
from loguru import logger
import torch

# %% 生成数据集


def synthetic_data(
        w: torch.Tensor,
        b: Union[float, torch.Tensor],
        num_examples: int, ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data for linear regression.
    Parameters:
    - w (torch.Tensor): The weight tensor.
    - b (Union[float, torch.Tensor]): The bias tensor or float value.
    - num_examples (int): The number of examples to generate.
    Returns:
    - tuple[Tensor, Tensor]: The features and labels.
    """
    dim = w.shape[0]
    x_shape = (num_examples, dim)
    # 从高斯分布中生成特征
    x = torch.normal(0, 1, x_shape)
    # 生成标签
    y = torch.matmul(x, w) + b
    # 添加噪声
    y += torch.normal(0, 0.01, y.shape)

    return x, y


def data_iter(batch_size: int, features: torch.Tensor, labels: torch.Tensor) -> Generator:
    """
    pass
    """
    # 获取数据集大小
    num_examples = features.shape[0]
    # 生成随机索引
    indices = list(range(num_examples))
    # 打乱索引
    random.shuffle(indices)

    # 逐渐生成 batch_size 大小的数据
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:i+batch_size])
        yield features[batch_indices], labels[batch_indices]

# %% 定义线性回归模型


def linear_regression(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs linear regression on the input data.
    Args:
        x (torch.Tensor): The input data tensor.
        w (torch.Tensor): The weight tensor.
        b (torch.Tensor): The bias tensor.
    Returns:
        torch.Tensor: The output tensor after performing linear regression.
    """
    return torch.matmul(x, w) + b

# %% 定义损失函数: 均方误差


def squared_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate the squared loss.
    Args:
        y_hat (torch.Tensor): The predicted values.
        y (torch.Tensor): The true values.
    Returns:
        torch.Tensor: The squared loss.
    """
    # 这里y.reshape 是因为 y_hat 和 y 的形状可能不一样, 比如 y_hat 是 (n, 1), y 是 (n,)
    delta = y_hat - y.reshape(y_hat.shape)
    return torch.sum(delta ** 2) / 2


# %% 定义优化算法: 小批量随机梯度下降
def batch_sgd(params: Tuple[torch.Tensor], lr: float, batch_size: int):
    """
    Mini-batch stochastic gradient descent.
    Args:
        params (Tuple[torch.Tensor, torch.Tensor]): The weight and bias tensors.
        lr (float): The learning rate.
        batch_size (int): The batch size.
    """
    with torch.no_grad():
        for p in params:    # 遍历所有参数
            p -= lr * p.grad / batch_size   # 更新参数, 这里除以 batch_size 是为了计算单个样本的梯度均值
            p.grad.zero_()  # 梯度清零

# %% 训练函数


def train(
        features: torch.Tensor,
        labels: torch.Tensor,
        net: Callable,
        loss: Callable,
        optimizer: Callable,
        batch_size: int = 16,
        num_epochs: int = 10,
        lr: float = 0.1):
    """
    Train a linear regression model.
    Args:
        features (torch.Tensor): The input features.
        labels (torch.Tensor): The labels.
        net (Callable): The network function.
        batch_size (int): The batch size.
        num_epochs (int): The number of epochs.
        lr (float): The learning rate.
    """
    # 1. 随机初始化模型参数
    # 权重, requires_grad=True 表示需要求梯度
    w = torch.normal(0, 0.01, (2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)  # 偏置, 同理需要求梯度

    for epoch in range(num_epochs):
        for batch_data in data_iter(batch_size, features, labels):
            # 1. 获取数据
            batch_features, batch_labels = batch_data
            # 2. 前向传播
            y_hat = net(batch_features, w, b)
            # 3. 计算损失
            loss_value = loss(y_hat, batch_labels)
            # 4. 反向传播
            loss_value.backward()
            # 5. 更新参数
            optimizer((w, b), lr, batch_size)
        logger.info(f'epoch {epoch + 1}, loss {loss_value:.4f}')

    return w, b


def main():
    """
    主函数
    """
    # 定义超参数
    num_examples = 1000
    batch_size = 16
    num_epochs = 10
    lr = 0.03

    # 生成合成数据集
    # 定义真实参数
    true_w = torch.tensor([2, -3.4])
    true_b = torch.tensor(4.2)
    features, labels = synthetic_data(true_w, true_b, num_examples)

    # 训练模型
    w_hat, b_bat = train(
        features,
        labels,
        linear_regression,
        squared_loss,
        batch_sgd,
        batch_size,
        num_epochs,
        lr)
    logger.info(f'\nw_hat: {w_hat.detach()}, \nb_hat: {b_bat.detach()}')
    logger.info(f'\ntrue_w: {true_w}, \ntrue_b: {true_b}')


# %%
if __name__ == '__main__':
    main()
# %%
