#%%
import torch
from torch import nn

# Parameters
# nn.Parameter 是 PyTorch 中的一个类，用于定义神经网络中的可训练参数。它是 torch.Tensor 的子类，但与普通张量不同的是，当它被分配为 nn.Module 的属性时，它会自动被添加到模型的参数列表中，并且会在调用 model.parameters() 时返回。
nn.Parameter(torch.tensor([1., 2., 3.]))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(2, 2))
        self.b = torch.tensor([1., 2.])
    
    def forward(self, x):
        return torch.matmul(x, self.W) + self.b

net = Net()
# 没有打印出 b，因为它不是 nn.Parameter 类型
for name, param in net.named_parameters():
    print(name, param)

#%% container
nn.Module()
nn.Sequential()
nn.ModuleList()
nn.ModuleDict()
nn.ParameterList()
nn.ParameterDict()

#%% hook
inputs = torch.tensor([[1., 2.], [3., 4.]])
net = nn.Linear(2, 2)
def hook_forward_pre(module, inputs):
    print("forward_pre hook")

def hook_forward(module, inputs, outputs):
    print("forward hook")
# 分别在前向传播前和前向传播后注册钩子
net.register_forward_pre_hook(hook_forward_pre)
net.register_forward_hook(hook_forward)
net(inputs)

# 类似的, 还有 backward_pre_hook 和 backward_hook


# %% 预定义的层
# Transformer
transformer = nn.Transformer(
    d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048
)
transformer(torch.rand(10, 32, 512), torch.rand(20, 32, 512))

# %%
