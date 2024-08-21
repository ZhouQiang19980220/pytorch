"""
工具类
"""
import os
import sys
from typing import Union
import torch
from torch import nn
from loguru import logger
import pandas as pd


def set_log_level(level=Union[str, int]) -> None:
    """
    设置显示的日志级别
    Args:
        level: str or int, 日志级别，可选值为 ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] 或 0-4
    Returns:
        None
    """
    levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if isinstance(level, str):
        level = level.upper()
        assert level in levels, f'level must be one of {levels}'
    elif isinstance(level, int):
        assert 0 <= level < len(levels)
        level = levels[level]
    logger.remove()
    logger.add(sys.stdout, level=level)


def all_exists(data_dir, *files):
    """
    检查文件是否存在
    Args:
        data_dir: str, 数据目录
        files: list, 文件列表
    Returns:
        bool, 如果所有文件都存在，则返回 True，否则返回 False
    """
    for file in files:
        if not os.path.exists(os.path.join(data_dir, file)):
            return False
    return True


def normalize(data):
    """
    标准化数据
    Args:
        data: pd.Series, 数据
    Returns:
        pd.Series, 标准化后的数据, 均值为 0，标准差为 1
    """
    return (data - data.mean()) / data.std()


def init_weights(m: nn.Module):
    """
    初始化权重
    Args:
        m: nn.Module, 神经网络层
    Returns:
        None
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m,
                    (
                        nn.Dropout,
                        nn.ReLU,
                        nn.Sequential,
                        nn.ModuleList,
                        nn.ModuleDict,
                    )):  # 跳过这些层, 因为这些层没有权重
        pass
    else:
        raise NotImplementedError(f'{m.__class__.__name__} not implemented')


def try_gpu(i: int = 0):
    """
    尝试返回 GPU(i), 如果不存在则返回 CPU
    Args:
        i: int, GPU 的索引
    Returns:
        torch.device, GPU 或 CPU
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def plot_loss(
    ax,
    training_losses: list[float],
    valid_losses: list[float]
) -> None:
    """
    绘制训练损失和验证损失
    Args:
        training_losses: list, 训练损失
        valid_losses: list, 验证损失
    Returns:
        None
    """
    ax.plot(training_losses, label='training loss')
    ax.plot(valid_losses, label='valid loss')
    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('Training and Validation Loss')


def save_submission(predictions: torch.Tensor, test_data: pd.DataFrame, file_name: str):
    """
    Save the submission file.
    """
    test_data['SalePrice'] = pd.Series(
        predictions.reshape(-1), index=test_data.index)
    submission = test_data[['Id', 'SalePrice']]
    submission.to_csv(file_name, index=False)

# 自动参数搜索
