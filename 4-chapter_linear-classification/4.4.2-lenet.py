# %%
from torch import nn
from utils import train_mnist, init_weights


class lenet(nn.Module):
    def __init__(self):
        super(lenet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.BatchNorm2d(6), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.ReLU(),
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


# %%
if __name__ == '__main__':
    main()

# %%
