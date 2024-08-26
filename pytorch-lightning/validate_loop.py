import torch
from torch import nn, optim, utils
from torchvision import transforms, datasets
import pytorch_lightning as L

#%% define a image classification model
class MLP(nn.Module):

    def __init__(
        self, 
        in_features: int = 784,
        out_features: int = 10,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, out_features)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#%% define a lightning module
class LitClassifier(L.LightningModule):

    def __init__(self, nn_classifier: nn.Module):
        super().__init__()
        self.model = nn_classifier

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', (y_hat.argmax(1) == y).float().mean())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', (y_hat.argmax(1) == y).float().mean())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', (y_hat.argmax(1) == y).float().mean())
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

#%% 
def main():
    training_data = datasets.MNIST(
        root='../data', train=True, download=True, transform=transforms.ToTensor())
    training_data_size = len(training_data)
    validation_data_size = int(0.2 * training_data_size)
    training_data_size = training_data_size - validation_data_size
    training_data, validation_data = utils.data.random_split(
        training_data, [training_data_size, validation_data_size])

    test_data = datasets.MNIST(
        root='../data', train=False, download=True, transform=transforms.ToTensor())

    num_workers = 16
    train_loader = utils.data.DataLoader(training_data, batch_size=64, shuffle=True, num_workers=num_workers)
    valid_loader = utils.data.DataLoader(validation_data, batch_size=64, num_workers=num_workers)
    test_loader = utils.data.DataLoader(test_data, batch_size=64, num_workers=num_workers)
    
    litClassifier = LitClassifier(MLP())
    trainer = L.Trainer(max_epochs=3)
    trainer.fit(litClassifier, train_loader, valid_loader)
    trainer.test(litClassifier, test_loader)

if __name__ == '__main__':
    main()