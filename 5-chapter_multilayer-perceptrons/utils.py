# %% import libraries
import os
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from matplotlib import pyplot as plt
from d2l import torch as d2l

def now():
    """
    获取当前时间
    """
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def get_worker():
    """
    如果是 Linux，则使用多进程加速
    """
    if os.name == 'posix':
        return 4
    return 0

def init_weights(m):
    """
    初始化权重
    """
    if isinstance(m, nn.Linear):    # 线性层
        nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):  # 卷积层
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# %% 训练


def train(
    net: nn.Module,
    dataloaders: dict,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int, 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    net.to(device)
    training_losses = []
    training_accuracies = []
    test_accuracies = []
    for epoch in range(num_epochs):
        # 训练循环
        net.train()
        training_loss = 0.0
        training_accuracy = 0
        training_num_samples = 0

        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)
            logits = net(images)
            loss_value = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            training_loss += loss_value.item()
            training_accuracy += d2l.accuracy(logits, labels)
            training_num_samples += labels.shape[0]

        # 测试循环
        net.eval()
        test_accuracy = 0.0
        test_num_samples = 0
        for images, labels in dataloaders['test']:
            images, labels = images.to(device), labels.to(device)
            logits = net(images)
            test_accuracy += d2l.accuracy(logits, labels)
            test_num_samples += labels.shape[0]

        # 保存并打印结果
        training_losses.append(training_loss / training_num_samples)
        training_accuracies.append(training_accuracy / training_num_samples)
        test_accuracies.append(test_accuracy / test_num_samples)
        print(f'-----epoch: {epoch + 1}-----')
        print(f'training loss: {training_losses[epoch]: .4f}')
        print(f'training accuracy: {training_accuracies[epoch]: .4f}')
        print(f'test accuracy: {test_accuracies[epoch]: .4f}')
    return training_losses, training_accuracies, test_accuracies
# %% 绘制 loss 和 accuracy 曲线


def train_mnist(net: nn.Module):
    # 设置参数
    batch_size = 256
    lr = 0.05
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 打印参数信息
    # 1. 操作系统, torch 版本, cuda 版本
    print(f'os: {os.name}')
    print(f'torch version: {torch.__version__}')
    print(f'cuda version: {torch.version.cuda}')
    # 2. 打印设备
    print(f'device: {device}')
    # 3. 打印参数
    print(f'batch size: {batch_size}')
    print(f'learning rate: {lr}')
    print(f'num epochs: {num_epochs}')
    # 4. 打印网络结构
    print(net)
    # 5. 打印开始时间
    print('training start at ', now())

    # 加载数据集
    trans = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    }

    datasets = {
        p: FashionMNIST(
            root='../data', train=(p == 'train'), transform=trans[p], download=True
        )
        for p in ['train', 'test']
    }

    dataloaders = {
        p: DataLoader(
            datasets[p], batch_size=batch_size, shuffle=(p == 'train'), num_workers=get_worker())
        for p in ['train', 'test']
    }

    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # 训练
    training_losses, training_accuracies, test_accuracies = train(
        net, dataloaders, loss_fn, optimizer, num_epochs, device=device
    )

    print('training end at ', now())

    # 绘制 loss 和 accuracy 曲线
    fig, ax = plt.subplots()
    training_loss_line,  = ax.plot(
        training_losses, label='training loss', color='red')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')

    ax2 = ax.twinx()
    training_accuracy_line, = ax2.plot(
        training_accuracies, label='training accuracy', color='red', linestyle='--')
    test_accuracy_line, = ax2.plot(
        test_accuracies, label='test accuracy', color='green', linestyle='--')
    ax2.set_ylabel('accuracy')

    lines = [training_loss_line, training_accuracy_line, test_accuracy_line]
    labels = [line.get_label() for line in lines]
    fig.legend(lines, labels, loc='center right')
    plt.show()
