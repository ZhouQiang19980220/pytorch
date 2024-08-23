"""
Resnet的实现
"""

#%% import libs
import unittest
from loguru import logger
import torch
from torch import nn

from d2l import torch as d2l

#%% 定义卷积块
class ConvBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # 注意这里的bias=False, 因为即便设置了bias=True, BatchNorm也会把bias消除掉
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

#%% 定义残差块
class BasicBlock(nn.Module):
    
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # stride只对第一个卷积层有效, 第二个卷积层的stride固定为1
        self.conv1 = ConvBatchNormRelu(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvBatchNormRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return x + identity

class BottleNeck(nn.Module):
    """
    BottleNeck块, 用于ResNet-50及其以上的网络
    1. 实际的卷积层的输出通道数是out_channels*expansion
    2. 中间的卷积层的stride设置为stride, 其他的stride都设置为1
    3. 1x1conv -> 3x3conv -> 1x1conv
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super().__init__()
        self.conv1 = ConvBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBatchNormRelu(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv3 = ConvBatchNormRelu(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return x + identity

#%% 定义ResNet
class ResNet(nn.Module):
    """
    ResNet的实现
    """
    configs = {
        "ResNet18":  ((2, 64), (2, 128), (2, 256),  (2, 512)), 
        "ResNet34":  ((3, 64), (4, 128), (6, 256),  (3, 512)),
        "ResNet50":  ((3, 64), (4, 128), (6, 256),  (3, 512)),
        "ResNet101": ((3, 64), (4, 128), (23, 256), (3, 512)),
        "ResNet152": ((3, 64), (8, 128), (36, 256), (3, 512))
    }
    def __init__(self, block, config, in_channels, num_classes):
        super().__init__()
        self.in_channels = 64
        # 每个stage都宽高减半, 通道数翻倍
        self.stage1 = nn.Sequential(
            ConvBatchNormRelu(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))   # 112x112
        self.stage2 = self._make_layer(block, config[0][1], config[0][0], stride=1) # 56x56
        self.stage3 = self._make_layer(block, config[1][1], config[1][0], stride=2) # 28x28
        self.stage4 = self._make_layer(block, config[2][1], config[2][0], stride=2) # 14x14
        self.stage5 = self._make_layer(block, config[3][1], config[3][0], stride=2) # 7x7

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(config[3][1]*block.expansion, num_classes)


    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None   # 下采样层用来改变通道或者宽高, 以便于与identity相加
        if stride != 1 or self.in_channels != out_channels*block.expansion:
            downsample = nn.Sequential(
                # 1x1conv, stride=stride来改变宽高, out_channels*block.expansion来改变通道
                nn.Conv2d(self.in_channels, out_channels*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*block.expansion))           
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels*block.expansion
        for _ in range(1, num_blocks):
            # 后面几个block的stride都设置为1, 不再进行下采样。
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNet18(ResNet):
    def __init__(self, in_channels, num_classes):
        super().__init__(BasicBlock, ResNet.configs["ResNet18"], in_channels, num_classes)
    
class ResNet34(ResNet):
    def __init__(self, in_channels, num_classes):
        super().__init__(BasicBlock, ResNet.configs["ResNet34"], in_channels, num_classes)

class ResNet50(ResNet):
    def __init__(self, in_channels, num_classes):
        super().__init__(BottleNeck, ResNet.configs["ResNet50"], in_channels, num_classes)

class ResNet101(ResNet):
    def __init__(self, in_channels, num_classes):
        super().__init__(BottleNeck, ResNet.configs["ResNet101"], in_channels, num_classes)

class ResNet152(ResNet):
    def __init__(self, in_channels, num_classes):
        super().__init__(BottleNeck, ResNet.configs["ResNet152"], in_channels, num_classes)

class MiNiResNet(ResNet):
    configs = {
        "MiNiResNet18":  ((2, 16), (2, 32), (2, 64),  (2, 128)), 
        "MiNiResNet34":  ((3, 16), (4, 32), (6, 64),  (3, 128)),
        "MiNiResNet50":  ((3, 16), (4, 32), (6, 64),  (3, 128)),
        "MiNiResNet101": ((3, 16), (4, 32), (23, 64), (3, 128)),
        "MiNiResNet152": ((3, 16), (8, 32), (36, 64), (3, 128))
    }

    def __init__(self, block, config, in_channels, num_classes):
        super().__init__(block, config, in_channels, num_classes)

class MiNiResNet18(MiNiResNet):
    def __init__(self, in_channels, num_classes):
        super().__init__(BasicBlock, MiNiResNet.configs["MiNiResNet18"], in_channels, num_classes)

class MiNiResNet34(MiNiResNet):
    def __init__(self, in_channels, num_classes):
        super().__init__(BasicBlock, MiNiResNet.configs["MiNiResNet34"], in_channels, num_classes)

class MiNiResNet50(MiNiResNet):
    def __init__(self, in_channels, num_classes):
        super().__init__(BottleNeck, MiNiResNet.configs["MiNiResNet50"], in_channels, num_classes)

class MiNiResNet101(MiNiResNet):
    def __init__(self, in_channels, num_classes):
        super().__init__(BottleNeck, MiNiResNet.configs["MiNiResNet101"], in_channels, num_classes)

class MiNiResNet152(MiNiResNet):
    def __init__(self, in_channels, num_classes):
        super().__init__(BottleNeck, MiNiResNet.configs["MiNiResNet152"], in_channels, num_classes)

class D2lClassifier(d2l.Classifier):
    def __init__(self, net, lr, num_classes):
        super().__init__()
        self.net = net
        self.lr = lr
        self.num_classes = num_classes

    def forward(self, x):
        return self.net(x)

#%% 测试ResNet
class TestResNet(unittest.TestCase):

    batch_size = 2
    in_features = 3
    num_classes = 10
    x = torch.randn(batch_size, in_features, 224, 224)

    def _test_resnet(self, resnet=ResNet18):
        in_features = self.in_features
        num_classes = self.num_classes
        model = resnet(in_features, num_classes)
        y = model(self.x)
        logger.info(model)
        logger.info(y.shape)
        self.assertEqual(y.shape, torch.Size([self.batch_size, self.num_classes]))

    def test_resnet18(self):
        logger.info("test_resnet18")
        self._test_resnet(ResNet18)

    def test_resnet34(self):
        logger.info("test_resnet34")
        self._test_resnet(ResNet34)
    
    def test_resnet50(self):
        logger.info("test_resnet50")
        self._test_resnet(ResNet50)

    def test_resnet101(self):
        logger.info("test_resnet151")
        self._test_resnet(ResNet101)

    def test_resnet152(self):
        logger.info("test_resnet152")
        self._test_resnet(ResNet152)

    def test_mini_resnet18(self):
        logger.info("test_mini_resnet18")
        self._test_resnet(MiNiResNet18)

    def test_mini_resnet34(self):
        logger.info("test_mini_resnet34")
        self._test_resnet(MiNiResNet34)

    def test_mini_resnet50(self):
        logger.info("test_mini_resnet50")
        self._test_resnet(MiNiResNet50)

    def test_mini_resnet101(self):
        logger.info("test_mini_resnet101")
        self._test_resnet(MiNiResNet101)

    def test_mini_resnet152(self):
        logger.info("test_mini_resnet152")
        self._test_resnet(MiNiResNet152)

#%%
def main():
    # unittest.main()

    model = miniresnet18 = D2lClassifier(MiNiResNet18(1, 10), 0.1, 10)
    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=16, resize=(224, 224))
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer.fit(model, data)


if __name__ == "__main__":
    main()

#%%
pass
