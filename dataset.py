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
