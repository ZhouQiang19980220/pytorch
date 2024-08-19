
#%%
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from matplotlib import pyplot as plt
from d2l import torch as d2l


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train(
    net: nn.Module,
    dataloaders: dict,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int
):
    training_losses = []
    training_accuracies = []
    test_accuracies = []
    for epoch in range(num_epochs):
        net.train()
        training_loss = 0.0
        training_accuracy = 0
        training_num_samples = 0

        for images, labels in dataloaders['train']:
            logits = net(images)
            loss_value = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            training_loss += loss_value.item()
            training_accuracy += d2l.accuracy(logits, labels)
            training_num_samples += labels.shape[0]

        net.eval()
        test_accuracy = 0.0
        test_num_samples = 0
        with torch.no_grad():
            for images, labels in dataloaders['test']:
                logits = net(images)
                test_accuracy += d2l.accuracy(logits, labels)
                test_num_samples += labels.shape[0]

        training_losses.append(training_loss / len(dataloaders['train']))
        training_accuracies.append(training_accuracy / training_num_samples)
        test_accuracies.append(test_accuracy / test_num_samples)
        print(f'-----epoch: {epoch + 1}-----')
        print(f'training loss: {training_losses[epoch]: .4f}')
        print(f'training accuracy: {training_accuracies[epoch]: .4f}')
        print(f'test accuracy: {test_accuracies[epoch]: .4f}')
    return training_losses, training_accuracies, test_accuracies


def train_mnist(net: nn.Module):
    batch_size = 256
    lr = 0.01
    num_epochs = 10

    trans = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    }

    datasets = {
        p: FashionMNIST(
            root='../data', train=(p == 'train'), transform=trans[p], download=True
        )
        for p in ['train', 'test']
    }

    dataloaders = {
        p: DataLoader(
            datasets[p], batch_size=batch_size, shuffle=(p == 'train'))
        for p in ['train', 'test']
    }

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    training_losses, training_accuracies, test_accuracies = train(
        net, dataloaders, loss_fn, optimizer, num_epochs
    )

    fig, ax = plt.subplots()
    training_loss_line,  = ax.plot(
        training_losses, label='training loss', color='red')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')

    ax2 = ax.twinx()
    training_accuracy_line, = ax2.plot(
        training_accuracies, label='training accuracy', color='red', linestyle='--')
    test_accuracy_line, = ax2.plot(
        test_accuracies, label='test accuracy', color='green', linestyle='--')
    ax2.set_ylabel('accuracy')

    lines = [training_loss_line, training_accuracy_line, test_accuracy_line]
    labels = [line.get_label() for line in lines]
    fig.legend(lines, labels, loc='center right')
    plt.show()


class lenet(nn.Module):
    def __init__(self):
        super(lenet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        feature = self.conv(x)
        x = self.flatten(feature)
        output = self.fc(x)
        return output


def main():
    net = lenet()
    net.apply(init_weights)
    train_mnist(net)


#%%
if __name__ == '__main__':
    main()
