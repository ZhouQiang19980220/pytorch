# %% import libraries
import torch
from typing import Union, Sequence, Optional
from torch import nn
from loguru import logger
import matplotlib.pyplot as plt

from dataset import HousePriceDataset
from models import HousePriceMLP
from loss import RelativeHousePriceLoss
from utils import set_log_level, try_gpu, init_weights, plot_loss


def train_epoch(
    net: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
):
    """
    训练一个 epoch
    Args:
        net: nn.Module, 神经网络
        dataloader: torch.utils.data.DataLoader, 数据加载器
        loss_fn: nn.Module, 损失函数
        optimizer: torch.optim.Optimizer, 优化器
        device: torch.device, 设备(GPU 或 CPU)
    Returns:
        float, 单个样本的平均损失
    """
    net.to(device)
    net.train()
    training_loss = 0.0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        outputs = net(features)
        loss = loss_fn(outputs, labels)
        training_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    training_loss /= len(dataloader)
    return training_loss

# 验证模型


def valid_epoch(
    net: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device
):
    """
    验证模型
    Args:
        net: nn.Module, 神经网络
        dataloader: torch.utils.data.DataLoader, 数据加载器
        loss_fn: nn.Module, 损失函数
        device: torch.device, 设备(GPU 或 CPU)
    Returns:
        float, 单个样本的平均损失
    """
    net.to(device)
    net.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = net(features)
            loss = loss_fn(outputs, labels)
            valid_loss += loss.item()
        valid_loss /= len(dataloader)
        return valid_loss


def train(
    net: nn.Module,
    training_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    log_interval: Optional[int] = None,

) -> list[list, list]:
    """
    训练模型
    Args:
        net: nn.Module, 神经网络
        training_dataloader: torch.utils.data.DataLoader, 训练数据加载器
        valid_dataloader: torch.utils.data.DataLoader, 验证数据加载器
        loss_fn: nn.Module, 损失函数
        optimizer: torch.optim.Optimizer, 优化器
        device: torch.device, 设备(GPU 或 CPU)
        num_epochs: int, 训练的轮数
    Returns:
        list[list, list], 两个列表，第一个列表为训练损失，第二个列表为验证损失
    """
    if log_interval is None:
        log_interval = num_epochs // 10
    # 开始训练
    training_losses = []
    valid_losses = []
    net.to(device)
    for epoch in range(num_epochs):
        training_loss = train_epoch(
            net, training_dataloader, loss_fn, optimizer, device)
        valid_loss = valid_epoch(
            net, valid_dataloader, loss_fn, device)
        training_losses.append(training_loss)
        valid_losses.append(valid_loss)
        # 必要的时候打印日志，显示误差
        if (epoch + 1) % log_interval == 0:
            logger.info(
                f'[epoch: {epoch+1:3d}/{num_epochs}], [training_loss: {training_loss:.4f}], [valid_loss: {valid_loss:.4f}]')
    return training_losses, valid_losses

# 定义一个k折交叉验证函数


def k_fold_cross_valid(
    k: int,
    net: nn.Module,
    training_dataset: HousePriceDataset,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 64,
    num_epochs: int = 100,
    log_interval: Optional[int] = None,
) -> list[list[list, list]]:
    """
    k 折交叉验证
    Args:
        k: int, k 折交叉验证的折数
        net: nn.Module, 神经网络
        training_dataset: HousePriceDataset, 训练数据集
        loss_fn: nn.Module, 损失函数
        optimizer: torch.optim.Optimizer, 优化器
        device: torch.device, 设备(GPU 或 CPU)
        num_epochs: int, 训练的轮数
        log_interval: int, 日志打印间隔
    Return:
        list[list[list, list]], 每一折的训练损失和验证损失
        return[0][0][0] 表示第 0 折的, 第 0 个 epoch 的训练损失
        return[0][1][0] 表示第 0 折的, 第 0 个 epoch 的验证损失
    """
    assert k > 1, 'k must be larger than 1'
    k_fold_training_losses = []
    k_fold_valid_losses = []
    # 存储初始化的网络参数和优化器参数
    net.to(device)
    for i, fold_data in enumerate(training_dataset.get_k_fold_data(k)):
        # 恢复网络参数和优化器参数
        net.net.apply(init_weights)
        # optimizer.load_state_dict(init_optimizer_state_dict)
        logger.info(f'[{i+1:2d}/{k:2d}] fold cross validation')
        training_features, training_labels, valid_features, valid_labels = fold_data
        logger.debug(
            f'{training_features.shape=}, {valid_features.shape=}')
        training_dataset = torch.utils.data.TensorDataset(
            training_features, training_labels)
        valid_dataset = torch.utils.data.TensorDataset(
            valid_features, valid_labels)
        training_dataloader = torch.utils.data.DataLoader(
            training_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False)
        training_losses, valid_losses = train(
            net,
            training_dataloader,
            valid_dataloader,
            loss_fn,
            optimizer,
            device,
            num_epochs=num_epochs,
            log_interval=log_interval)
        k_fold_training_losses.append(training_losses)
        k_fold_valid_losses.append(valid_losses)
        logger.info(f'[{i+1:2d}/{k:2d}] fold cross validation done')
    return k_fold_training_losses, k_fold_valid_losses


# %%
if __name__ == '__main__':
    def main():
        """
        pass
        """
        # set_log_level('INFO')
        set_log_level('DEBUG')
        device = try_gpu()
        logger.info(f'Using device: {device}')

        # 定义超参数
        num_epochs = 100
        batch_size = 64
        learning_rate = 0.01
        k = 5
        weight_decay = 1e-5

        # 日志打印间隔
        log_interval = 10

        # 加载数据
        training_dataset = HousePriceDataset()

        # 定义网络, 并查看网络结构
        net = HousePriceMLP()
        net.to(device)
        logger.debug(f'net = \n{net}')

        # 损失函数
        loss_fn = RelativeHousePriceLoss()

        # 优化器
        optimizer = torch.optim.Adam(
            net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # k 折交叉验证
        k_fold_training_losses, k_fold_valid_losses = k_fold_cross_valid(
            k,
            net,
            training_dataset,
            loss_fn,
            optimizer,
            device,
            batch_size=batch_size,
            num_epochs=num_epochs,
            log_interval=log_interval
        )
        # 保存模型
        torch.save(net.state_dict(), 'house_price_mlp.pth')
        # 打印每一折的最后一个 epoch 的训练损失和验证损失
        for i, (training_losses, valid_losses) in enumerate(zip(k_fold_training_losses, k_fold_valid_losses)):
            logger.info(
                f'[{i+1:2d}/{k:2d}] fold cross validation, [training_loss: {training_losses[-1]:.4f}], [valid_loss: {valid_losses[-1]:.4f}]')

        # 打印所有折的平均训练损失和验证损失
        avg_training_losses = torch.tensor(k_fold_training_losses).mean(dim=0)
        avg_valid_losses = torch.tensor(k_fold_valid_losses).mean(dim=0)
        logger.info(
            f'[{k: 2d}] fold cross validation, [avg_training_loss: {avg_training_losses[-1]: .4f}], [avg_valid_loss: {avg_valid_losses[-1]: .4f}]')

        for i in range(k):
            fig, ax = plt.subplots()
            plot_loss(ax, k_fold_training_losses[i], k_fold_valid_losses[i])
        plt.show()
    main()

# %%
