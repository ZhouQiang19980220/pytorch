# %% import libraries
import torch
from torch import nn
import pandas as pd
from loguru import logger

from models import HousePriceMLP
from dataset import HousePriceDataset
from utils import save_submission, set_log_level


def predict(
    net: nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
) -> torch.Tensor:
    """
    Predict house prices using a trained model.
    """
    net.eval()
    predictions = []
    with torch.no_grad():
        for features in test_dataloader:
            predictions.append(net(features))
    return torch.cat(predictions, dim=0).reshape(-1, 1)


def main():
    set_log_level('INFO')
    # set_log_level('DEBUG')

    test_dataset = HousePriceDataset(is_train=False)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False
    )

    net = HousePriceMLP()
    net.load_state_dict(torch.load('house_price_mlp.pth'))
    logger.info('predicting ...')
    predictions = predict(net, test_dataloader)
    save_submission(
        predictions,
        test_dataset.raw_test_data,
        file_name='submission.csv'
    )
    logger.info('saved submission.csv')


if __name__ == '__main__':
    main()

#%%