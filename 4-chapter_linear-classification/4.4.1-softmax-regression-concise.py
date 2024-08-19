from torch import nn
from utils import train_mnist, init_weights


# %%
if __name__ == '__main__':
    def main():
        net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 10))
        net.apply(init_weights)
        train_mnist(net)
    main()
# %%
