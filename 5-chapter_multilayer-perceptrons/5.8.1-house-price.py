#%% import libraries
import os
import sys
import torch
import pandas as pd
from loguru import logger
from d2l import torch as d2l
#%% 设置显示的日志级别
def set_log_level(level='INFO'):
    logger.remove()
    logger.add(sys.stdout, level=level)
level = 'INFO'
set_log_level(level)
#%% load dataset
data_dir = '../data'
train_file = 'kaggle_house_pred_train.csv'
test_file = 'kaggle_house_pred_test.csv'
def all_exists(data_dir, **files):
    for file in files:
        if not os.path.exists(os.path.join(data_dir, file)):
            return False
    return True
download = not all_exists(data_dir, *[train_file, test_file])   # 如果数据集不存在，则下载
if download:    # 下载数据集
    logger.info('Downloading data...')
    raw_train_data = pd.read_csv(
        d2l.download(d2l.DATA_URL + train_file, data_dir)
    )
    raw_test_data = pd.read_csv(
        d2l.download(d2l.DATA_URL + test_file, data_dir)
    )
else:   # 使用本地数据集
    logger.info('Using local data...')
    raw_train_data = pd.read_csv(os.path.join(data_dir, train_file))
    raw_test_data = pd.read_csv(os.path.join(data_dir, test_file))
logger.debug(f'{raw_train_data.shape=}')
logger.debug(f'{raw_test_data.shape=}')
# 查看一些数据
logger.debug(
    # 0: Id, (不能作为特征)
    # 1: MSSubClass, 
    # 2: MSZoning, 
    # 3: LotFrontage, 
    # -3: SaleType,
    # -2: SaleCondition,
    # -1: SalePrice
    raw_train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]
)

# %% 数据预处理
logger.info('Preprocessing data...')
# 1. 删除第 1 列（Id）
train_data = raw_train_data.iloc[:, 1:-1]  # 训练数据的最后一列是 SalePrice, 作为标签，暂时删除
test_data = raw_test_data.iloc[:, 1:]   # 测试数据没有 SalePrice, 因此不需要删除

# 2. 将所有的训练数据和测试数据连接起来
all_features = pd.concat((train_data, test_data))

# 3. 对连续数值的特征做标准化处理: 均值为 0, 标准差为 1
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 计算均值和标准差
mean = all_features[numeric_features].mean()
std = all_features[numeric_features].std()
logger.debug(f'\nmean: {mean}')
logger.debug(f'\nstd: {std}')
# 标准化
all_features[numeric_features] = (all_features[numeric_features] - mean) / std
# 把缺失值填充为 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 4. 处理离散数值特征: one-hot 编码
all_features = pd.get_dummies(all_features, dummy_na=True)
logger.debug(f'{all_features.shape=}')
#%%
# 将数据转换为张量
logger.info('Converting data to tensor...')
n_train = train_data.shape[0]
train_features = torch.tensor(
    all_features[:n_train].values.astype(float), dtype=torch.float32
    )
test_features = torch.tensor(
    all_features[n_train:].values.astype(float), dtype=torch.float32
    )

train_labels = torch.tensor(
    raw_train_data.SalePrice.values.astype(float), dtype=torch.float32
    )

#%%


