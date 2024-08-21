"""
vgg
"""
# %% import libraries
from loguru import logger
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

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


if __name__ == "__main__":
    # test_vgg11()
    # 在 Fashion MNIST 数据集上训练和测试 MiniVGG11
    # 0. 定义超参数
    batch_size = 64
    num_epochs = 10
    optimizer_name = 'SGD'
    optimizer_params = {
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 5e-4
    }
    model_name = 'MiniVGG11'
    model_params = {
        'in_channels': 1,
        'num_classes': 10
    }
    device = get_device()
    logger.info(f'device: {device}')

    # 1. 获取数据集
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
        p: DataLoader(dataset[p], batch_size=batch_size,
                      shuffle=(p == 'train'))
        for p in train_phases
    }

    # 2. 定义模型
    if model_name == 'MiniVGG11':
        model = MiniVGG11(**model_params)
    else:
        raise ValueError(f'Unsupported model: {model_name}')
    model.to(device)

    # 3. 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 4. 定义优化器
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')

    # 5. 训练模型
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
                    running_corrects += torch.sum(preds == labels.data)
                    running_counts += inputs.size(0)
        logger.info(
            get_log_string(
                epoch,
                num_epochs,
                phase,
                running_loss / running_counts,
                running_corrects / running_counts)
        )
