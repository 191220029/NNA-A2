import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class DataSet(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        if self.labels is not None:
            return sample, self.labels[idx]
        return sample

class CSVDataSet(DataSet):
    def __init__(self, csv_file, label_col=None, transform=None):
        data_frame = pd.read_csv(csv_file)
        if label_col is not None:
            labels = data_frame[label_col].values
            data = data_frame.drop(columns=[label_col]).values
        else:
            labels = None
            data = data_frame.values
        super().__init__(data, labels, transform)

class MNISTDataSet(DataSet):
    def __init__(self, root, train=True, transform=None, download=True):
        dataset = datasets.MNIST(root=root, train=train, transform=transform, download=download)
        self.data = dataset.data.numpy()
        self.labels = dataset.targets.numpy()
        self.transform = transform

class FashionMNISTDataSet(DataSet):
    def __init__(self, root, train=True, transform=None, download=True):
        dataset = datasets.FashionMNIST(root=root, train=train, transform=transform, download=download)
        self.data = dataset.data.numpy()
        self.labels = dataset.targets.numpy()
        self.transform = transform

class DataLoaderWrapper:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

# 使用示例：
# Iris 数据集
iris_dataset = CSVDataSet("data/iris.csv", label_col="species")
iris_loader = DataLoaderWrapper(iris_dataset, batch_size=32, shuffle=True)

# MNIST 数据集
mnist_dataset = MNISTDataSet(root="./data", train=True, transform=transforms.ToTensor(), download=True)
mnist_loader = DataLoaderWrapper(mnist_dataset, batch_size=32, shuffle=True)

# Fashion-MNIST 数据集
fashion_mnist_dataset = FashionMNISTDataSet(root="./data", train=True, transform=transforms.ToTensor(), download=True)
fashion_mnist_loader = DataLoaderWrapper(fashion_mnist_dataset, batch_size=32, shuffle=True)