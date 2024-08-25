import torch
from torch import nn
from d2l import torch as d2l

#%% 合成时序数据: 正弦函数与加性噪声
T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time)
noise = torch.normal(mean=0, std=0.2, size=(T,))
x += noise
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

# 任务: 假设已知过去的数据，预测接下来的  τ  个时间步
# 马尔科夫假设: x(t)  仅依赖于  x(t-1), x(t-2), ..., x(t-τ) 的信息
tau = 4
# TODO: 这里改进一下，不要用循环
# 构建特征矩阵, 每行是一个样本, 每列是一个特征
features = torch.zeros(size=(T - tau, tau))
for i in range(tau):
    features[:, i] = x[i:T - tau + i]   # 第 i 列是 x(t-i)
# 构建标签
labels = x[tau:].reshape((-1, 1))
batch_size = 16
n_train = 600
# 转换为迭代器
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 平方损失: 用于回归模型
loss = nn.MSELoss()

# 训练
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')
net = get_net()
train(net, train_iter, loss, 5, 0.01)
#%% 查看预测结果
onestep_preds = net(features)
d2l.plot(
    [time, time[tau:]], 
    [x.detach().numpy(), onestep_preds.detach().numpy()], 
    'time', 'x', 
    legend=['data', '1-step preds'], 
    xlim=[1, 1000], figsize=(6, 3)) 

#%% 进行多步预测
# 多步预测效果很差, 因为误差一直在累积
multistep_preds = torch.zeros(T)
multistep_preds[:n_train + tau] = x[:n_train + tau] # 训练集上做多步预测
for i in range(n_train + tau, T):
    # 前面 tau 个数据是已知的都是预测值
    multistep_preds[i] = net(multistep_preds[i - tau:i].reshape((1, -1)))
d2l.plot(
    [time, time[tau:], time[n_train + tau:]], 
    [x.detach().numpy(), onestep_preds.detach().numpy(), multistep_preds[n_train + tau:].detach().numpy()], 
    'time', 'x', 
    legend=['data', '1-step preds', 'multistep preds'], 
    xlim=[1, 1000], figsize=(6, 3)
)
#%% 分别进行(1, 4, 16, 64)步预测
# TODO: 后续完成固定步数多步预测
steps = (1, 4, 16, 64)

