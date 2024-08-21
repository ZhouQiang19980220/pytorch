import os
from typing import Generator
import torch
import pandas as pd
from loguru import logger
from utils import all_exists
from d2l import torch as d2l

from utils import normalize


class HousePriceDataset(torch.utils.data.Dataset):
    """
    房价预测数据集
    """

    def __init__(
        self,
        data_dir: str = '../../data',
        training_file: str = 'kaggle_house_pred_train.csv',
        test_file: str = 'kaggle_house_pred_test.csv',
        is_train: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.training_file = training_file
        self.test_file = test_file
        self.is_train = is_train
        self._load_data()   # 加载数据
        self._preprocess_data()

    def _load_data(self, ):
        data_dir = self.data_dir
        training_file = self.training_file
        test_file = self.test_file
        self.download = not all_exists(data_dir, *[training_file, test_file])
        if self.download:
            logger.info('Downloading data...')
            self.raw_training_data = pd.read_csv(
                d2l.download(d2l.DATA_URL + training_file, data_dir)
            )
            self.raw_test_data = pd.read_csv(
                d2l.download(d2l.DATA_URL + test_file, data_dir)
            )
        else:
            logger.info('Using local data...')
            self.raw_training_data = pd.read_csv(
                os.path.join(data_dir, training_file))
            self.raw_test_data = pd.read_csv(
                os.path.join(data_dir, test_file))

    def _preprocess_data(self, ):
        logger.info('Preprocessing data...')
        label = 'SalePrice'
        features = pd.concat((
            self.raw_training_data.drop(columns=['Id', label]),
            self.raw_test_data.drop(columns=['Id'])
        ))
        numeric_features = features.dtypes[features.dtypes != 'object'].index
        features[numeric_features] = \
            features[numeric_features].apply(normalize)
        features[numeric_features] = features[numeric_features].fillna(0)
        features = pd.get_dummies(features, dummy_na=True)

        n_train = self.raw_training_data.shape[0]
        self.training_features = torch.tensor(
            features[:n_train].values.astype(float), dtype=torch.float32)
        self.test_features = torch.tensor(
            features[n_train:].values.astype(float), dtype=torch.float32)
        self.training_labels = torch.tensor(
            self.raw_training_data.SalePrice.values.astype(float), dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        if self.is_train:
            return self.training_features.shape[0]
        else:
            return self.test_features.shape[0]

    def __getitem__(self, idx):
        if self.is_train:
            return self.training_features[idx], self.training_labels[idx]
        else:
            return self.test_features[idx]

    def get_k_fold_data(
        self,
        k: int = 5,
    ) -> Generator:
        """
        获取 k 折交叉验证的数据
        """
        # 验证 k 的合法性
        assert k > 1, 'k must be larger than 1'
        # 计算每一折的大小, 也就是验证集的大小
        fold_size = self.training_features.shape[0] // k
        slices = []
        for i in range(k):
            start = i * fold_size
            end = (i + 1) * fold_size
            if end > self.training_features.shape[0]:
                end = self.training_features.shape[0]
            slices.append(slice(start, end))

        for i in range(k):
            valid_slice = slices[i]
            valid_features = self.training_features[valid_slice]
            valid_labels = self.training_labels[valid_slice]
            training_features = torch.cat(
                [self.training_features[s] for j, s in enumerate(slices) if j != i], dim=0)
            training_labels = torch.cat(
                [self.training_labels[s] for j, s in enumerate(slices) if j != i], dim=0)
            yield training_features, training_labels, valid_features, valid_labels
