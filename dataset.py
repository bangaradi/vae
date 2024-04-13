import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

class MNIST_Custom(Dataset):
    def __init__(self, digits, data_path, train=True, transform=None, download=True):
        self.digits = digits
        self.data_path = data_path
        self.train = train
        self.transform = transform

        # Load MNIST dataset
        if self.train:
            self.dataset = datasets.MNIST(root=self.data_path, train=True, transform=None, download=download)
        else:
            self.dataset = datasets.MNIST(root=self.data_path, train=False, transform=None, download=False)

        # Filter dataset to include only specified digits
        self.indices = [i for i, (image, label) in enumerate(self.dataset) if label in digits]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, label = self.dataset[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class MNISTWithNoise(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.noise_label = 10  # Label for noise images
        self.total_mnist = len(mnist_dataset)
        self.noise_labels_length = self.total_mnist // 10

    def __len__(self):
        # Total length is MNIST images plus one noise image per label
        return self.total_mnist + self.noise_labels_length

    def __getitem__(self, idx):
        if idx < self.total_mnist:
            data, label = self.mnist_dataset[idx]
        else:
            # Generate a noise image
            data = torch.randn(1, 28, 28)
            label = self.noise_label
        return data, label

# # Define the digits you want to include
# selected_digits = [0, 1, 2]

# # Define your data path
# data_path = './data'

# # Create a custom dataset
# custom_train_dataset = CustomDigitsDataset(digits=selected_digits, data_path=data_path, train=True, transform=transforms.ToTensor())
# custom_test_dataset = CustomDigitsDataset(digits=selected_digits, data_path=data_path, train=False, transform=transforms.ToTensor())

# # Create data loaders
# batch_size = 64
# train_loader = DataLoader(dataset=custom_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# test_loader = DataLoader(dataset=custom_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
