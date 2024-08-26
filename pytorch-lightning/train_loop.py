# 学习pytorch-lightning的基本流程
 
#%% import libraries
import os
import torch
from torch import nn, optim, utils
from torchvision import transforms, datasets
import lightning as L

from loguru import logger

#%% define the model
class AutoEncoder(nn.Module):

    def __init__(
        self, 
        in_features: int = 784,
    ):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(in_features, 64), 
            nn.ReLU(),
            nn.Linear(64, 3),
        )

        self.decode = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, in_features),
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

#%% define a lightning module
class LitAutoEncoder(L.LightningModule):
    
    def __init__(self, nn_autoencoder: nn.Module):
        super().__init__()
        self.model = nn_autoencoder
    
    def training_step(self, batch, batch_idx):
        """
        training step defines the train loop, it is independent of forward
        """
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    # TODO: set float32 matmul precision
    torch.set_float32_matmul_precision('medium')
    lit_autoencoder = LitAutoEncoder(AutoEncoder())
    training_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
    training_dataloader = utils.data.DataLoader(training_dataset, batch_size=32, shuffle=True, num_workers=47)

    trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(lit_autoencoder, training_dataloader)


