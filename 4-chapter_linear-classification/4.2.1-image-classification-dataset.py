# %% import libraries
from typing import List
import torch
import torchvision

from d2l import torch as d2l

# %%
# % matplotlib inline
# %% download the dataset
trans = torchvision.transforms.ToTensor()
# 训练集
mnist_train = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True)
# 测试集
mnist_test = torchvision.datasets.FashionMNIST(
    root='../data', train=False, transform=trans, download=True)

# %% data loadaer
batch_size = 16
workers = 0
train_iter = torch.utils.data.DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, num_workers=workers)
test_iter = torch.utils.data.DataLoader(
    mnist_test, batch_size=batch_size, shuffle=False, num_workers=workers)

# 打印训练集和测试集的样本数
print(f'训练集样本数: {len(mnist_train)}, 测试集样本数: {len(mnist_test)}')
print(f'batch_size: {batch_size}')
first_batch = next(iter(train_iter))
batch_images, batch_labels = first_batch
print(
    f'images shape: {batch_images.shape}, labels shape: {batch_labels.shape}')
# %% show the data


def get_fashion_mnist_labels(labels: torch.Tensor) -> List[str]:
    """
    Get the text label for the Fashion-MNIST dataset.
    Args:
        labels (torch.Tensor): The labels.
    Returns:
        List[str]: The text labels.
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


print(f'the first batch of labels: {get_fashion_mnist_labels(batch_labels)}')

# %%


def show_images(
        images: torch.Tensor,
        num_rows: int,
        num_cols: int,
        titles: List[str] = None,
        scale: float = 1.5,
) -> None:
    """
    Plot a list of images.
    Args:
        images (torch.Tensor): The images.
        num_rows (int): The number of rows.
        num_cols (int): The number of columns.
        titles (List[str]): The titles.
        scale (float): The scale of the images.
    Returns:
        None
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (image, ax) in enumerate(zip(images, axes)):
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        ax.imshow(image)    # 显示图像
        ax.axes.get_xaxis().set_visible(False)  # 隐藏 x 轴
        ax.axes.get_yaxis().set_visible(False)  # 隐藏 y 轴
        if titles:  # 如果有标签, 则设置标题
            ax.set_title(titles[i])
    return axes


show_images(
    images=batch_images.permute(0, 2, 3, 1),
    num_rows=4,
    num_cols=4,
    titles=get_fashion_mnist_labels(batch_labels)
)

# %% 统计遍历数据集所需时间
timer = d2l.Timer()
for x, y in train_iter:
    continue
f'遍历训练集所需时间: {timer.stop():.2f} sec'   # 3.06s
# %% 统计数据集中每个类别的样本数
label_counts = torch.zeros(10, dtype=torch.int32)
for _, labels in train_iter:
    # bincount: 统计每个类别的样本数
    label_counts += torch.bincount(labels, minlength=10)
for i in range(10):
    print(f'{get_fashion_mnist_labels([i])[0]}: {label_counts[i]}')
# %%
