import scipy
import torch
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
import torchvision.transforms as transforms
from sklearn.datasets import make_regression, make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.nn as nn
import torchvision
from sklearn.datasets import make_moons, make_circles, make_swiss_roll
import sklearn
import pandas as pd
from ucimlrepo import fetch_ucirepo
from torchvision.datasets import ImageFolder
import os
import numpy as np
from math import ceil, pi

from torch_geometric.datasets import KarateClub, Planetoid

# Some predefined paths
MNIST_data_path = '/xxxx/data/'
CIFAR_data_path = '/xxxx/data/'
HOUSE_data_path = '/xxxx/data/'
TINYIMAGE_data_path = '/xxxx/data/'
SVHN_data_path = '/xxxx/data/'
CIFAR10_data_path = '/xxxx/data/'

def gen_toy_dataset(n_samples, n_features, test_size=0.25):
    # Generate synthetic data for regression
    X, y = make_friedman1(n_samples, n_features, noise=0.0, random_state=42)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Normalize data using StandardScaler after splitting
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler(feature_range=(0, 1))  # Separate scaler for y
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()  # Normalize and reshape y
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for targets
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    return train_dataset, test_dataset


def generate_testbench(test_bench, batch_size, num_workers):
    train_loader, test_loader, criterion, process_func = None, None, None, None
    if test_bench == 'toyregress':
        n_samples, n_features, test_size = 4096, 5, 0.40
        train_dataset, test_dataset = gen_toy_dataset(n_samples, n_features, test_size)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        criterion = nn.MSELoss()
        process_func = lambda x: x
    elif test_bench == 'housing':
        data = sklearn.datasets.fetch_california_housing(data_home=HOUSE_data_path)
        x = data['data']
        y = data['target'].reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        scaler_x, scaler_y = MinMaxScaler(feature_range=(0, 1)).fit(X_train), MinMaxScaler().fit(y_train)
        X_train, X_test = scaler_x.transform(X_train), scaler_x.transform(X_test)
        y_train, y_test = scaler_y.transform(y_train), scaler_y.transform(y_test)

        # dataloader
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                     torch.tensor(y_test, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        criterion = nn.MSELoss()
        process_func = lambda x: x
    elif test_bench == 'diabete':
        data = sklearn.datasets.load_diabetes()
        x = data['data']
        y = data['target'].reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        scaler_x, scaler_y = MinMaxScaler(feature_range=(0, 1)).fit(X_train), MinMaxScaler(feature_range=(0, 1)).fit(
            y_train)
        X_train, X_test = scaler_x.transform(X_train), scaler_x.transform(X_test)
        y_train, y_test = scaler_y.transform(y_train), scaler_y.transform(y_test)

        # dataloader
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                     torch.tensor(y_test, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        criterion = nn.MSELoss()
        process_func = lambda x: x

    elif test_bench == 'mnist':
        # load in mnist data from torchvision
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        train_dataset = torchvision.datasets.MNIST(root=MNIST_data_path, train=True, transform=transforms.Compose(
            [transforms.RandomRotation(degrees=15),  # Randomly rotate by up to 15 degrees
             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate by up to 10% of image size
             transforms.ToTensor(), ]), download=True)


        test_dataset = torchvision.datasets.MNIST(root=MNIST_data_path, train=False,
                                                  transform=transforms.Compose([transforms.ToTensor(), ]),
                                                  download=True)
        # data loader
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # loss function
        criterion = nn.CrossEntropyLoss()
        # process function
        process_func = lambda x: (x[0].reshape(x[0].shape[0], -1), x[1])

    elif test_bench == 'cifar100':

        train_dataset = torchvision.datasets.CIFAR100(root=CIFAR_data_path, train=True,
                                                      download=False, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
                transforms.ToTensor()
            ]))
        test_dataset = torchvision.datasets.CIFAR100(root=CIFAR_data_path, train=False,
                                                     download=False,
                                                     transform=transforms.Compose([transforms.ToTensor(), ]))

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # loss function
        criterion = nn.CrossEntropyLoss()
        # process function
        process_func = lambda x: (x[0].reshape(x[0].shape[0], -1), x[1])
    elif test_bench == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root=CIFAR10_data_path, train=True,
                                                     download=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
                transforms.ToTensor()
            ]))
        test_dataset = torchvision.datasets.CIFAR10(root=CIFAR10_data_path, train=False,
                                                    download=True,
                                                    transform=transforms.Compose([transforms.ToTensor(), ]))
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # loss function
        criterion = nn.CrossEntropyLoss()
        # process function
        process_func = lambda x: (x[0].reshape(x[0].shape[0], -1), x[1])
    elif test_bench == 'svhn':
        train_dataset = torchvision.datasets.SVHN(root=SVHN_data_path, split='train',
                                                  download=True, transform=transforms.Compose([
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
                transforms.ToTensor()
            ]))
        test_dataset = torchvision.datasets.SVHN(root=SVHN_data_path, split='test',
                                                    download=True,
                                                    transform=transforms.Compose([transforms.ToTensor(), ]))
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # loss function
        criterion = nn.CrossEntropyLoss()
        # process function
        process_func = lambda x: (x[0].reshape(x[0].shape[0], -1), x[1])
    elif test_bench == 'tinyimage':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
        ])

        # Create ImageFolder datasets for train and test sets
        train_dataset = ImageFolder(os.path.join(TINYIMAGE_data_path, 'train/'), transform=transform)
        test_dataset = ImageFolder(os.path.join(TINYIMAGE_data_path, 'val/'), transform=transform)

        # Create data loaders for train and test datasets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        # Define loss function
        criterion = nn.CrossEntropyLoss()

        # Define process function
        process_func = lambda x: (x[0].reshape(x[0].shape[0], -1), x[1])
    elif test_bench == 'circle':
        points, _ = make_circles(n_samples=4096, noise=0.06, factor=0.5)
        x = TensorDataset(torch.tensor(points).type(torch.float32))
        train_loader = DataLoader(x, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        process_func = lambda x: x[0]
    elif test_bench == 'twomoon':
        points, _ = make_moons(n_samples=4096, noise=0.1)
        x = TensorDataset(torch.tensor(points).type(torch.float32))
        train_loader = DataLoader(x, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        process_func = lambda x: x[0]
    elif test_bench == 'genmnist':
        # load in mnist data from torchvision
        train_dataset = torchvision.datasets.MNIST(root=MNIST_data_path, train=True, transform=transforms.Compose([transforms.ToTensor(), ]), download=True)
        selected_labels = [2]
        selected_indices = [i for i in range(len(train_dataset)) if train_dataset.targets[i] in selected_labels]
        subset_dataset = torch.utils.data.Subset(train_dataset, selected_indices)
        print(f"Number of data in subset_dataset: {len(subset_dataset)}")
        train_loader = DataLoader(dataset=subset_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        process_func = lambda x: x[0].reshape(x[0].shape[0], -1)
    elif test_bench == 'gauss':
        x = TensorDataset((torch.randn(4096, 2) - 1.0).type(torch.float32))
        train_loader = DataLoader(x, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        process_func = lambda x: x[0]
    elif test_bench == 'swissroll':
        points, _ = make_swiss_roll(n_samples=4096, noise=1.0)
        points = points[:, [0, 2]] / 5.0
        x = TensorDataset(torch.tensor(points).type(torch.float32))
        train_loader = DataLoader(x, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        process_func = lambda x: x[0]
    elif test_bench == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        x = TensorDataset(torch.tensor(x).type(torch.float32))
        train_loader = DataLoader(x, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        process_func = lambda x: x[0]
    elif test_bench == "pinwheel":
        # from: https://github.com/google-research/google-research/blob/master/aloe/aloe/common/synthetic/toy_data_gen.py
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = np.random.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        x = 2 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))
        x = TensorDataset(torch.tensor(x).type(torch.float32))
        train_loader = DataLoader(x, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        process_func = lambda x: x[0]
    elif test_bench == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        x = TensorDataset(torch.tensor(dataset).type(torch.float32))
        train_loader = DataLoader(x, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        process_func = lambda x: x[0]
    else:
        raise NotImplementedError(f"Test bench {test_bench} is not implemented")

    return train_loader, test_loader, criterion, process_func


# https://arxiv.org/pdf/1505.05770.pdf
class PotentialFunc(object):
    def __init__(self, name: str):
        self.potential = getattr(self, name)

    def __call__(self, z: torch.Tensor, cal_type=1) -> torch.Tensor:
        if cal_type == 1:
            if z.shape[1] != 2:
                raise ValueError(f"Input shape {z.shape} is not supported")
            else:
                return self.potential(z)
        else:
            raise NotImplementedError(f"Cal type {cal_type} is not implemented")

    def potential1(self, z: torch.Tensor) -> torch.Tensor:
        z1, z2 = z[:, 0], z[:, 1]
        t1 = 0.5 * ((torch.norm(z, dim=1) - 2) / 0.4) ** 2
        wrk1 = torch.exp(-0.5 * ((z1 - 2) / 0.6) ** 2)
        wrk2 = torch.exp(-0.5 * ((z1 + 2) / 0.6) ** 2)
        t2 = torch.log(wrk1 + wrk2)
        return t1 - t2

    def potential2(self, z: torch.Tensor) -> torch.Tensor:
        z1, z2 = z[:, 0], z[:, 1]
        w1 = torch.sin(2 * np.pi * z1 / 4)
        return 0.5 * ((z2 - w1) / 0.4) ** 2