"""
常见的 pytorch API
"""
#%%
import torch

#%%
torch.is_tensor(torch.tensor([1, 2]))   # True
# %%
# 如果 input 是一个单元素张量，并且在类型转换后不等于零，则返回 True。
torch.is_nonzero(torch.tensor([0, ]))   # False
# %%
torch.is_nonzero(torch.tensor([1, ]))   # True
#%%
# Boolean value of Tensor with more than one value is ambiguous
try:
    torch.is_nonzero(torch.tensor([1, 2]))
except RuntimeError as e:
    print(e) 
# %%
# 返回 input 张量中的元素总数。
torch.numel(torch.tensor([[1, 2], [3, 4]]))   # 4

# %% 创建张量
# 从 numpy 数组创建张量
import numpy as np
a = np.array([1, 2, 3])
torch.from_numpy(a)
# %%
# 创建全 0 或者全 1 的张量
torch.zeros(size=(2, 3)), torch.ones(size=(2, 3)), torch.zeros_like(torch.tensor([[1, 2], [3, 4]])), torch.ones_like(torch.tensor([[1, 2], [3, 4]]))
# %% 创建顺序张量
torch.arange(0, 2.5, 0.5)

#%% linspace: 在(0, 10)之间均匀取11个点
torch.linspace(0, 10, 11)

# %% 单位矩阵
torch.eye(3)

#%% 空张量
torch.empty(size=(2, 3)), torch.empty_like(torch.eye(3))
# %% 填充指定值的张量
torch.full(size=(2, 3), fill_value=3), torch.full_like(torch.eye(3), fill_value=3)

#%% 复张量
torch.complex(real=torch.tensor([1., 2.]), imag=torch.tensor([3., 4.]))

# %% 索引
# %% argwhere
tensor = torch.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
#%% 返回非零元素的索引, 一共3个非零元素，返回3个索引
torch.argwhere(tensor), torch.nonzero(torch.eye(3))

#%% 连接
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])
torch.cat((x, y), dim=0), torch.cat((x, y), dim=1)

#%% cat 的别名
torch.concat((x, y), dim=0), torch.concat((x, y), dim=1)

#%% cat 的别名
torch.concatenate((x, y), dim=0), torch.concatenate((x, y), dim=1)
# %% 列堆叠
torch.column_stack((x, y)), torch.hstack((x, y))
# %% 行堆叠
torch.row_stack((x, y)), torch.vstack((x, y))

#%% 水平分割
torch.hsplit(x, 2), torch.split(x, 1, dim=1)

#%% 垂直分割
torch.vsplit(x, 2), torch.split(x, 1, dim=0)

#%% 调整张量形状
x = torch.tensor([[1, 2], [3, 4], [5, 6]])
x.reshape(2, 3), x.view(2, 3), x.reshape(-1, 2), x.view(-1, 2)
# %% 压缩维度
x = torch.zeros(size=(2, 1, 2, 1))                                
x.shape, torch.squeeze(x).shape, torch.squeeze(x, dim=1).shape

# %% 展开维度
x = torch.zeros(size=(2, 3))
x.shape, torch.unsqueeze(x, dim=0).shape, x[None, :, :].shape
# %% 交换轴
x = torch.arange(6).reshape(2, 3)
x.shape, x.t().shape, x.T.shape, x.transpose(0, 1).shape, x.swapaxes(0, 1).shape, x.swapdims(0, 1).shape
# %% 交换多个轴, 一般用于 torch 和 numpy 之间的转换
x = torch.arange(48).reshape(3, 4, 4)    # torch的格式: (channel, height, width)
x.shape, x.permute(1, 2, 0).shape        # numpy的格式: (height, width, channel)
# %%
# 条件选择
x = torch.arange(9).reshape(3, 3) - 5
x, torch.where(x>0, 1, -1) # 生成符号

# %% 关闭或者开启局部梯度计算
x = torch.tensor([1., 2.], requires_grad=True)
with torch.no_grad():   # 关闭局部梯度计算
    y = x * 2
y.requires_grad    # False
# %%
# 开启局部梯度计算
x = torch.tensor([1., 2.], requires_grad=True)
with torch.enable_grad():   # 开启局部梯度计算
    y = x * 2
y.requires_grad    # True
# %%
# 设置局部梯度计算
is_train = False
x = torch.tensor([1., 2.], requires_grad=True)
with torch.set_grad_enabled(is_train):   # 设置局部梯度计算
    y = x * 2
y.requires_grad    # False
# %% 数学运算
# 绝对值
x = torch.tensor([-1, -2, 3])
torch.abs(x), torch.absolute(x)

#%% 反余弦
x = torch.rand(size=(2, 2))
torch.acos(x), torch.arccos(x)

#%% 反正弦
x = torch.rand(size=(2, 2))
torch.asin(x), torch.arcsin(x)
# 类似的, 还有反正切, 反双曲正切, 反双曲余弦, 反双曲正弦
# 同理也可以计算正弦, 余弦, 正切, 双曲正切, 双曲余弦, 双曲正弦
#%%
# 加法: 将 other 乘以 alpha 后加到 input 上。
x = torch.tensor([1, 2])
torch.add(x, other=torch.ones_like(x), alpha=2), x + 2
#%% 逐元素除法
x = torch.tensor([1, 3])
torch.div(x, other=torch.ones_like(x) * 2), torch.divide(x, other=torch.ones_like(x) * 2), x / 2

#%% 去整
x = torch.tensor([1.2, 2.51, 3.9])
# 向上取整, 向下取整， 四舍五入
torch.ceil(x), torch.floor(x), torch.round(x)

#%% 向上或者向下截断
x = torch.tensor([1.2, 2.51, 3.9])
torch.clamp(x, min=2, max=3), torch.clip(x, min=2, max=3)
# %% 计算 tensor 的小数部分
x = torch.tensor([1.2, 2.51, 3.9])
torch.frac(x)

# %% 将 nan 替换为指定值
x = torch.tensor([1., float('nan'), 3., float('inf'), float('-inf')])
torch.nan_to_num(x, nan=0, posinf=1, neginf=-1)
# %% 将 tensor 转换为 Python list 或者 numpy array
tensor = torch.tensor([1, 2, 3])
tensor.tolist(), tensor.numpy()

#%%