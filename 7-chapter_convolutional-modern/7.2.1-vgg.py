"""
vgg
"""
# %% import libraries
import os
from typing import Sequence, Optional
import warnings
from loguru import logger
import objprint
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import SGD

# %% define VGG


class VGGBlock(nn.Module):
    """
    VGG块
    1. 由若干个卷积层(包含 ReLU)和一个最大池化层组成
    2. 不改变输入的高和宽,  但是会改变通道数
    """

    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        layers.append(nn.ReLU())
        for _ in range(num_convs - 1):
            layers.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGG(nn.Module):
    """
    VGG模型
    1. 由若干个VGG块组成
    2. 使用config来指定每个VGG块的超参数
    """

    # 这里定义全部支持的配置
    # 例如: VGG11, VGG13, VGG16, VGG19
    # 这里的11, 13, 16, 19表示卷积层数+全连接层数
    configs = {
        # (卷积层数, 输出通道数)
        "VGG11": ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
        "VGG13": ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512)),
        "VGG16": ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512)),
        "VGG19": ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512)),
    }

    def __init__(self, config, in_channels=3, num_classes=10):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.num_classes = num_classes
        self._build_model()

    def _build_model(self):
        layers = []
        in_channels = self.in_channels
        # 每个 Block 通道翻倍, 图片尺寸减半
        for num_convs, out_channels in self.config:
            layers.append(VGGBlock(in_channels, out_channels, num_convs))
            in_channels = out_channels
        layers.append(nn.Flatten())
        # 全连接层, 这里的7*7是因为输入图片的大小是224*224, 经过5个VGG块后(2^5=32 倍下采样), 图片大小变为7*7
        layers.append(nn.Linear(out_channels * 7 * 7, 4096))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(4096, 4096))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(4096, self.num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VGG11(VGG):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__(VGG.configs["VGG11"], in_channels, num_classes)


class VGG13(VGG):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__(VGG.configs["VGG13"], in_channels, num_classes)


class VGG16(VGG):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__(VGG.configs["VGG16"], in_channels, num_classes)


class VGG19(VGG):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__(VGG.configs["VGG19"], in_channels, num_classes)


class MiniVGG(VGG):
    configs = {
        # (卷积层数, 输出通道数)
        "VGG11": ((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)),
        "VGG13": ((2, 16), (2, 32), (2, 64), (2, 128), (2, 128)),
        "VGG16": ((2, 16), (2, 32), (3, 64), (3, 128), (3, 128)),
        "VGG19": ((2, 16), (2, 32), (4, 64), (4, 128), (4, 128)),
    }

    def __init__(self, config, in_channels=3, num_classes=10):
        super().__init__(config, in_channels, num_classes)


class MiniVGG11(MiniVGG):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__(MiniVGG.configs["VGG11"], in_channels, num_classes)


class MiniVGG13(MiniVGG):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__(MiniVGG.configs["VGG13"], in_channels, num_classes)


class MiniVGG16(MiniVGG):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__(MiniVGG.configs["VGG16"], in_channels, num_classes)


class MiniVGG19(MiniVGG):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__(MiniVGG.configs["VGG19"], in_channels, num_classes)


def forward_hook_print_shape(
    module: nn.Module,
    input: torch.Tensor,
    output: torch.Tensor
):
    """
    定义一个hook用于在forward时打印shape
    """
    print(f"{module.__class__.__name__}: {output.shape}")


def get_log_string(
    epoch: int,
    num_epochs: int,
    phase: str,
    loss: float,
    acc: float
) -> str:
    """
    获取日志字符串
    """
    epoch_str = f"[epoch: {epoch+1:3d}/{num_epochs:3d}]"
    phase_str = f"[{phase:>5}]"
    loss_str = f"[loss: {loss:.6f}]"
    acc_str = f"[acc: {acc:.4f}]"
    return f"{epoch_str}, {phase_str}, {loss_str}, {acc_str}"


def test_vgg11():
    """
    测试VGG11模型
    """
    model = VGG11()
    logger.info(f'model: \n{model}')
    for module in model.model:
        module.register_forward_hook(forward_hook_print_shape)
    x = torch.randn(1, 3, 224, 224)
    logger.info(f'input shape: {x.shape}')
    y = model(x)


def get_device():
    """
    获取设备
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_num_workers():
    """
    获取num_workers
    """
    if torch.cuda.is_available():
        return 4
    else:
        return 0

def get_model(
    model_name: str,
    model_params: dict,
):
    """
    根据模型名和参数获取模型
    """
    supported_models = (
        'VGG11', 'VGG13', 'VGG16', 'VGG19',
        'MiniVGG11', 'MiniVGG13', 'MiniVGG16', 'MiniVGG19'
    )
    if model_name not in supported_models:
        raise ValueError(f'Unsupported model: {model_name}')
    return eval(model_name)(**model_params)

def get_optimizer(
    optimizer_name: str,
    optimizer_params: dict,
    model: nn.Module
):
    """
    根据优化器名和参数获取优化器
    """
    supported_optimizers = ('SGD')
    if optimizer_name not in supported_optimizers:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')
    return eval(optimizer_name)(model.parameters(), **optimizer_params)

def get_fashion_mnist_dataset(
    batch_size: int,
    num_workers: int
):
    """
    获取Fashion MNIST数据集
    """
    train_phases = ['train', 'valid']
    transform = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),
        'valid': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
    }

    dataset = {
        p: datasets.FashionMNIST(
            root='../data', train=(p == 'train'), download=True, transform=transform[p])
        for p in train_phases
    }

    dataloader = {
        p: DataLoader(dataset[p], batch_size=batch_size, shuffle=(p == 'train'), num_workers=num_workers)
        for p in train_phases
    }

    return dataloader

def save_model(
    model: nn.Module,
    save_dir: str,
    save_name: str,
    force: bool = False
):
    """
    保存模型
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(os.path.join(save_dir, save_name)):
        if force:
            warnings.warn(f'{os.path.join(save_dir, save_name)} exists, but force is True, so overwrite it')
            torch.save(model.state_dict(), os.path.join(save_dir, save_name))
        else:
            warnings.warn(f'{os.path.join(save_dir, save_name)} exists, set force=True to overwrite it')
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, save_name))

class Metric():

    def __init__(
        self, 
        metric_names: Sequence[str],
    ):  
        if isinstance(metric_names, (str, Sequence)):
            if isinstance(metric_names, str):
                metric_names = [metric_names]
        else:
            raise TypeError(f'metric_names should be str or Sequence, but got {type(metric_names)}')
        
        self.metric_names = metric_names
        self.metrics = dict()
        for metric_name in metric_names:
            self.metrics[metric_name] = []

    def append(
        self, 
        metric_names: Sequence[str],
        metric
    ):
        for metric_name in metric_names:
            assert metric_name in self.metric_names, f'{metric_name} not in {self.metric_names}'
            self.metrics[metric_name].append(metric)

    def append_all(
        self, 
        metrics: dict
    ):
        for metric_name in self.metric_names:
            self.metrics[metric_name].append(metrics[metric_name])
    
    def get(self, metric_name: str):
        return self.metrics[metric_name]

    def get_all(self):
        return self.metrics

    def get_last(self, metric_name: str):
        return self.metrics[metric_name][-1]

    def get_last_all(self):
        return {metric_name: self.metrics[metric_name][-1] for metric_name in self.metric_names}

    def get_max(self, metric_name: str):
        max_value, max_index = max((value, index) for index, value in enumerate(self.metrics[metric_name]))
        return max_value, max_index

    def get_min(self, metric_name: str):
        min_value, min_index = min((value, index) for index, value in enumerate(self.metrics[metric_name]))
        return min_value, min_index

    def get_best(self, metric_name: str, good=True):
        if good:
            return self.get_max(metric_name)
        else:
            return self.get_min(metric_name)


    def __repr__(self):
        return objprint.obj2str(self.metrics)
    
    __str__ = __repr__


if __name__ == "__main__":
    # test_vgg11()
    # 在 Fashion MNIST 数据集上训练和测试 MiniVGG11
    # 定义训练配置
    train_config = dict(
        model_name = 'MiniVGG11', 
        model_params = {
            'in_channels': 1,
            'num_classes': 10
        }, 
        optimizer_name = 'SGD', 
        optimizer_params = {
            'lr': 0.005,
            'momentum': 0.9,
            'weight_decay': 5e-4
        },
        get_dataset = get_fashion_mnist_dataset, 
        get_dataset_params = {
            'batch_size': 64,
            'num_workers': get_num_workers()
        }, 
        num_epochs = 10,
        patience = 10,
        log_interval = 1, 
        device = get_device(), 
        save = False, 
        save_interval = 1,
        save_params = {
            'save_dir': '../models/mini_vgg11_fashion_mnist',
            'force': False
        }
    )
    logger.info(f'train_config: \n{objprint.objstr(train_config)}')

    def train(
        model_name: str,
        model_params: dict,
        optimizer_name: str,
        optimizer_params: dict,
        num_epochs: int,
        patience: int,
        log_interval: int,
        device: torch.device,
        get_dataset: callable,
        get_dataset_params: dict,
        save: bool,
        save_interval: Optional[int] = None,
        save_params: Optional[int] = None
    ):
        """
        训练模型
        Args:
            model_name: 模型名
            model_params: 模型参数
            optimizer_name: 优化器名
            optimizer_params: 优化器参数
            num_epochs: 训练轮数
            patience: int, 早停轮数
            log_interval: 日志间隔
            device: 设备
            get_dataset: 获取数据集的函数
            get_dataset_params: 获取数据
            save: bool, # 是否保存模型
            save_interval: Optional[int] = None,    # 保存间隔
            save_params: Optional[int] = None   # 保存参数
        """
        # 检查参数
        if save:
            assert save_interval is not None, 'save_interval should not be None when save is True'
            assert save_params is not None, 'save_params should not be None when save is True'
        
        # 初始配置
        train_phases = ['train', 'valid']
        loss_metric = Metric(
            metric_names = train_phases
        )
        acc_metric = Metric(
            metric_names = train_phases
        )

        # # 1. 获取数据集
        dataloader = get_dataset(**get_dataset_params)

        # # 2. 定义模型
        model = get_model(model_name, model_params)
        model.to(device)

        # # 3. 定义损失函数
        criterion = nn.CrossEntropyLoss()

        # # 4. 定义优化器
        optimizer = get_optimizer(optimizer_name, optimizer_params, model)
        
        best_acc = 0.0
        # # 5. 训练模型
        for epoch in range(num_epochs):
            for phase in train_phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                running_counts = 0
                for inputs, labels in dataloader[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data).item()
                        running_counts += inputs.size(0)
                loss_metric.append([phase], running_loss / running_counts)
                acc_metric.append([phase], running_corrects / running_counts)
                if phase == 'valid':
                    if running_corrects / running_counts > best_acc:
                        best_acc = running_corrects / running_counts
                        patience_count = 0
                    else:
                        patience_count += 1
                    if patience_count >= patience:
                        logger.info(f'early stopping at epoch {epoch+1}')
                        return loss_metric, acc_metric
                if (epoch+1) % log_interval == 0:
                    logger.info(
                        get_log_string(
                            epoch,
                            num_epochs,
                            phase,
                            running_loss / running_counts,
                            running_corrects / running_counts)
                    )
            if save and (epoch+1) % save_interval == 0:
                save_name = f'{model_name}_{epoch+1}.pth'
                save_model(model, save_name=save_name, **save_params)
                logger.info(f'saved model {save_name}')

        return loss_metric, acc_metric
    loss_metric, acc_metric = train(**train_config)
    
    # 打印最好的结果, 以val_acc 为准
    _, best_valid_acc_epoch = acc_metric.get_best('valid', good=True)
    logger.info(f'best results is at epoch {best_valid_acc_epoch}')
    logger.info(
        get_log_string(
            best_valid_acc_epoch,
            100, 
            'train',
            loss_metric.get('train')[best_valid_acc_epoch], 
            acc_metric.get('train')[best_valid_acc_epoch]
        )
    )
    logger.info(
        get_log_string(
            best_valid_acc_epoch,
            100,
            'valid',
            loss_metric.get('valid')[best_valid_acc_epoch], 
            acc_metric.get('valid')[best_valid_acc_epoch]
        )
    )