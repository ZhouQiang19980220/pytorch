# %% import librarys
import torch
import torchvision
from IPython import display
from d2l import torch as d2l
# %% load dataset
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# %% initialize model parameters
num_inputs = 28 * 28
num_outputs = 10

# 初始户参数
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs),
                 requires_grad=True)   # 784 * 10
b = torch.zeros(num_outputs, requires_grad=True)   # 10

# %% define softmax operation


def softmax(x: torch.Tensor):
    # x.shape = (batch_size, num_outputs)
    x_exp = torch.exp(x)
    # 这里keepdim=True是为了保持维度，便于广播
    partition = torch.sum(x_exp, dim=1, keepdim=True)
    return x_exp / partition


# 验证一下softmax函数
fake_outputs = torch.randn(batch_size, num_outputs)
fake_outputs_prob = softmax(fake_outputs)
# soft max函数的每一行求和都应该是 1
line_sum = torch.sum(fake_outputs_prob, dim=1)
# 由于浮点数的原因，这里不能直接用==1，只能用allclose
assert torch.allclose(line_sum, torch.ones(batch_size))
# 验证每一个元素都是在0-1之间
assert torch.all(fake_outputs_prob >= 0)
assert torch.all(fake_outputs_prob <= 1)
fake_outputs_prob.shape
# %% define model


def net(X: torch.Tensor):
    # X.shape = (batch_size, 1, 28, 28)
    X = X.reshape((-1, num_inputs))  # X.shape = (batch_size, 784)
    hidden = torch.matmul(X, W) + b  # linear operation
    return softmax(hidden)  # softmax operation

# %% define loss function 交叉熵


def cross_entropy(
    y_hat: torch.Tensor,
    y: torch.Tensor
):
    # y_hat.shape = (batch_size, num_outputs)
    # y.shape = (batch_size, )
    prob = y_hat[torch.arange(len(y_hat)), y]   # 列表索引, 取出每一行对应的标签
    return -torch.log(prob)


# 检验一下交叉熵
fake_y = torch.tensor([0, 2])
fake_y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
torch.allclose(cross_entropy(fake_y_hat, fake_y) +
               torch.log(torch.tensor([0.1, 0.5])), torch.tensor([0.0]))

# %% 计算准确样本的个数


def accuracy(y_hat: torch.Tensor, y: torch.Tensor):
    # y_hat.shape = (batch_size, num_outputs)
    # y.shape = (batch_size, )
    # 如果y_hat是二维的，那么就取出每一行的最大值的索引
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 比较结果
    return float(cmp.type(y.dtype).sum())


# 验证准确率
torch.allclose(accuracy(fake_y_hat, fake_y) /
               len(fake_y_hat), torch.tensor(0.5))
# %%
