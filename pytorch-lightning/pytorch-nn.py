# %% import libraries
from typing import Sequence, Union
from functools import partial
import math
import torch
from torch import nn
from torch.nn import functional as F
from loguru import logger
# %%
net = []
x = torch.randn(2, 20, dtype=torch.float32)
# %%
# Sequential model
net.append(
    nn.Sequential(
        nn.Linear(20, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
)
logger.debug(f'net = \n{net[-1]}')
# (2, 20) -> (2, 256) -> (2, 10)
logger.debug(f'{net[-1](x).shape=}')


# %% 自定义块
class MLP(nn.Module):   # 继承 nn.Module
    """
    多层感知机
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[Sequence[int], int],
        output_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_dim[0]))
        self.layers.append(nn.ReLU())
        for i in range(1, len(hidden_dim)):
            self.layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim[-1], output_dim))
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)


net.append(
    MLP(20, [256, 10], 10)
)
logger.debug(f'net = \n{net[-1]}')
# (2, 20) -> (2, 256) -> (2, 10) -> (2, 10)
logger.debug(f'{net[-1](x).shape=}')

# %% 自定义 Sequential


class MySequential(nn.Module):
    """
    自定义 Sequential
    """

    def __init__(self, *args):
        super().__init__()
        self.layers = nn.ModuleList(args)

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x


net.append(
    MySequential(
        nn.Linear(20, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
)
logger.debug(f'net = \n{net[-1]}')
# (2, 20) -> (2, 256) -> (2, 10)
logger.debug(f'{net[-1](x).shape=}')

# %% 参数管理
net.append(
    nn.Linear(2, 3)
)
logger.debug(f'net = \n{net[-1]}')
# 状态字典: OrderedDict; 有 2 个 key: weight, bias
logger.debug(f'net.state_dict() = \n{net[-1].state_dict()}')
logger.debug(f'net.state_dict().keys() = \n{net[-1].state_dict().keys()}')
# 从状态字典中分别打印出权重和偏置的形状
# weight: (3, 2); bias: (3,)
logger.debug(
    f'net.state_dict()["weight"].shape = \n{net[-1].state_dict()["weight"].shape}')
logger.debug(
    f'net.state_dict()["bias"].shape = \n{net[-1].state_dict()["bias"].shape}')
# 也可以通过名字来获取参数
logger.debug(f'net.weight.data.shape = \n{net[-1].weight.data.shape}')
logger.debug(f'net.bias.data.shape = \n{net[-1].bias.data.shape}')
# 也可以访问参数的梯度
# 由于还没有进行反向传播计算梯度，所以梯度是 None
logger.debug(f'net.weight.grad = \n{net[-1].weight.grad}')
logger.debug(f'net.bias.grad = \n{net[-1].bias.grad}')

# 遍历所有参数
# name_param 是一个元组，包含参数的名字和参数本身
for name, param in net[-1].named_parameters():
    logger.debug(f'{name}: {param.shape}')

# %% 初始化模型参数


def init_normal(
    m: nn.Module,
):
    """
    初始化模型参数, 使得权重服从正态分布
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m) in (nn.ReLU, ):
        pass
    elif type(m) in (nn.Sequential, ):
        pass
    else:
        raise ValueError(f'Unrecognized layer: {type(m)}')


def init_constant(
    m: nn.Module,
):
    """
    初始化模型参数, 使得权重为常数, 偏置为 0
    """
    # 这里只是演示，实际中不会将权重初始化为常数
    # 因为这样会导致反向传播计算梯度时梯度为常数
    # 相当于只有一个隐藏层节点, 因为所有节点的权重都是一样的, 梯度也都是一样的
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m) in (nn.ReLU, ):
        pass
    elif type(m) in (nn.Sequential, ):
        pass
    else:
        raise ValueError(f'Unrecognized layer: {type(m)}')


net.append(
    nn.Sequential(
        nn.Linear(20, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
)
net[-1].apply(init_constant)
assert torch.allclose(
    net[-1][0].weight.data, torch.ones_like(net[-1][0].weight.data))
logger.debug(f'{net[-1][0].weight.data[0]}')   # 全部为 1
# %% 参数初始化: 带参数
# 方法 1: 通过类实现
# 方法 2: 通过闭包实现
# 方法 3: 通过偏函数实现


class InitConstant():
    def __init__(self, val):
        self.val = val

    def __call__(self, m):
        if type(m) == nn.Linear:
            nn.init.constant_(m.weight, self.val)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif type(m) in (nn.ReLU, ):
            pass
        elif type(m) in (nn.Sequential, ):
            pass
        else:
            raise ValueError(f'Unrecognized layer: {type(m)}')


net.append(
    nn.Sequential(
        nn.Linear(20, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
)
net[-1].apply(InitConstant(2))
assert torch.allclose(
    net[-1][0].weight.data, 2.0*torch.ones_like(net[-1][0].weight.data))
logger.debug(f'{net[-1][0].weight.data[0]}')   # 全部为 2

# 闭包


def init_constant_val(val):
    """
    闭包实现参数初始化
    """
    def init(m):
        if type(m) == nn.Linear:
            nn.init.constant_(m.weight, val)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif type(m) in (nn.ReLU, ):
            pass
        elif type(m) in (nn.Sequential, ):
            pass
        else:
            raise ValueError(f'Unrecognized layer: {type(m)}')
    return init


net.append(
    nn.Sequential(
        nn.Linear(20, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
)
net[-1].apply(InitConstant(3))
assert torch.allclose(
    net[-1][0].weight.data, 3.0*torch.ones_like(net[-1][0].weight.data))
logger.debug(f'{net[-1][0].weight.data[0]}')   # 全部为 3

# 偏函数


def init_normal_mean_std(
    m: nn.Module,
    mean: float = 0,
    std: float = 0.01
):
    """
    初始化模型参数, 使得权重服从正态分布
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=mean, std=std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m) in (nn.ReLU, ):
        pass
    elif type(m) in (nn.Sequential, ):
        pass
    else:
        raise ValueError(f'Unrecognized layer: {type(m)}')


net.append(nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
))
# 使用偏函数
net[-1].apply(partial(init_normal_mean_std, mean=0, std=0.01))
logger.debug(f'使用偏函数初始化权重')

# 常见的实用初始化方法
# 1. Xavier 初始化
nn.init.xavier_normal_
nn.init.xavier_uniform_
# 2. Kaiming 初始化
nn.init.kaiming_normal_
nn.init.kaiming_uniform_
# 3. 正态分布初始化
nn.init.normal_
# 4. 均匀分布初始化
nn.init.uniform_
# 5. 常数初始化: 一般用于初始化偏置
nn.init.constant_
# 5.1 全 0 初始化
nn.init.zeros_
# 5.2 全 1 初始化
nn.init.ones_
# 6. 单位矩阵初始化
nn.init.eye_
# 7. 正交初始化
nn.init.orthogonal_
# 8. 稀疏初始化
nn.init.sparse_
# 9. 狄拉克初始化
nn.init.dirac_

# %% 自定义初始化函数


def my_init_constant_(tensor, val):
    """
    自定义初始化函数
    """
    with torch.no_grad():
        tensor.fill_(val)


def my_init(m):
    if type(m) == nn.Linear:
        my_init_constant_(m.weight, 1)
        if m.bias is not None:
            my_init_constant_(m.bias, 0)
    elif type(m) in (nn.ReLU, ):
        pass
    elif type(m) in (nn.Sequential, ):
        pass
    else:
        raise ValueError(f'Unrecognized layer: {type(m)}')


net.append(nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
))
net[-1].apply(my_init)
assert torch.allclose(
    net[-1][0].weight.data, torch.ones_like(net[-1][0].weight.data))
logger.debug(f'{net[-1][0].weight.data}')   # 全部为 1

# %% 参数绑定
# 有时我们希望在多个层之间共享模型参数
# Module 类的 forward 函数里多次调用同一个层
shared = nn.Linear(8, 8)
net.append(nn.Sequential(
    nn.Linear(20, 8),
    nn.ReLU(),
    shared,
    nn.ReLU(),
    shared,
    nn.ReLU(),
    nn.Linear(8, 10)
))
logger.debug('参数绑定')
# 第 2 层和第 4 层共享参数
assert net[-1][2].weight.data.eq(net[-1][4].weight.data).all()
net[-1][2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
assert net[-1][2].weight.data[0, 0] == net[-1][4].weight.data[0, 0]
# 事实上，他们的内存地址是一样的
assert id(net[-1][2].weight) == id(net[-1][4].weight)

# %% 自定义层: 减去均值, 除以标准差


class Norm(nn.Module):
    """
    自定义层: 减去均值, 除以标准差
    这个层不包含模型参数
    """

    def __init__(
            self,
            eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        # x.shape = (batch_size, num_features)
        mean = x.mean(dim=0, keepdim=True)

        # 求标准差时加上一个很小的常数 eps，避免除以 0
        # 这里的标准差使用有偏估计，即方差的分母是 n 而不是 n-1，为了对齐 nn.BatchNorm1d
        std = x.std(dim=0, keepdim=True, unbiased=False) + self.eps
        return (x - mean) / std


net.append(
    Norm()
)
logger.debug(f'net = \n{net[-1]}')
# (2, 20) -> (2, 20)
logger.debug(f'{net[-1](x).shape=}')
# 检验一下这个层真的标准化了数据
y1 = net[-1](x)
net.append(
    nn.BatchNorm1d(20, affine=False, momentum=1.0)
)
y2 = net[-1](x)


y3 = F.batch_norm(
    x,
    running_mean=x.mean(dim=0),
    running_var=x.var(dim=0, unbiased=False),
    weight=None,
    bias=None,
)
y1[0, :5], y2[0, :5], y3[0, :5]
# %% 自定义线性层


class MyLinear(nn.Module):
    """
    自定义线性层
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:   # 注意这里, 如果不含偏置，则需要注册 bias 为 None
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self, ):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in + 1e-6)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        outputs = torch.matmul(
            x, self.weight.T
        )
        if self.bias is not None:
            outputs += self.bias
        return outputs

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


# 验证一下自定义线性层和 PyTorch 自带的 nn.Linear 是否一致
net.append(
    nn.Linear(20, 10)
)
net.append(
    MyLinear(20, 10)
)
net[-1].weight.data = net[-2].weight.data
net[-1].bias.data = net[-2].bias.data
assert torch.allclose(
    net[-1](x), net[-2](x)
)

# 测试一下反向传播
net.append(
    MyLinear(20, 10)
)
y = torch.randn(2, 10)
y_hat = net[-1](x)
loss_fn = torch.nn.MSELoss()
loss_value = loss_fn(y_hat, y)
loss_value.backward()
logger.debug(f'{net[-1].weight.grad.shape=}')
logger.debug(f'{net[-1].bias.grad.shape=}')
# %% 自定义激活函数


class MyReLU(nn.Module):
    """
    自定义激活函数
    """

    def __init__(self, ):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.clamp(min=0)   # 截断小于 0 的值

# %% 自定义优化器


class MySGD(torch.optim.Optimizer):
    """
    自定义优化器
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
    ):
        defaults = dict(lr=lr)  # 默认参数
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:  # 遍历参数组
            for p in group['params']:   # 遍历参数
                if p.grad is None:
                    continue
                grad = p.grad
                # 注意，这里更新参数的方式是直接修改 p.data.add_, 而不是使用 p.add_
                # 这是因为 p 是一个 Parameter 对象，而 Parameter 对象是 Tensor 的子类
                # 如果直接使用 p.add_，则会导致参数更新的过程被记录到计算图中
                # 但是我们不希望将参数更新的过程记录到计算图中，因此使用 p.data.add_
                p.data.add_(-group['lr'] * grad)
