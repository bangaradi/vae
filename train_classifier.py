import torch
import torchvision
import torch.nn.functional as F
import os
from model import Classifier
from dataset import MNISTWithNoise
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./dataset", help="Path of MNIST dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Train batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=20, help='Number of epochs')
    args = parser.parse_args()
    return args


def train(epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optim.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optim.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
    
        train_losses.append(loss.item())
        train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
    torch.save(net.state_dict(), './classifier_ckpts/model.pt')
    
    
def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()
    # some other training parameters
    batch_size_test = 1000
    log_interval = 100

    os.makedirs("./classifier_ckpts", exist_ok=True)


    mnist_train = torchvision.datasets.MNIST(args.data_path, train=True, download=True,
                                            transform=torchvision.transforms.ToTensor())
    mnist_test = torchvision.datasets.MNIST(args.data_path, train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())

    # train_dataset = MNISTWithNoise(mnist_train)
    # test_dataset = MNISTWithNoise(mnist_test)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
                    torchvision.datasets.MNIST(args.data_path, train=True, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor()
                                                ])),
                                                batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                    torchvision.datasets.MNIST(args.data_path, train=False, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor()
                                                ])),
                                                batch_size=batch_size_test, shuffle=True)
    


    net = Classifier().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(args.n_epochs + 1)]
    
    print("Starting training...")
    for epoch in range(1, args.n_epochs + 1):
        train(epoch)
        scheduler.step()
        test()

# from torchvision.datasets import ImageFolder
# import torchvision.transforms as transforms

# import torchvision.transforms as transforms
# from torchvision.datasets import MNIST
# from torch.utils.data import DataLoader, Dataset
# import numpy as np
# import os
# from PIL import Image

# test_directory = 'test'
# os.makedirs(test_directory, exist_ok=True)
# for i in range(11):  # For digits 0-9 and noise label 10
#     os.makedirs(os.path.join(test_directory, str(i)), exist_ok=True)

# class NoiseDataset(Dataset):
#     """ Dataset to generate Gaussian noise images """
#     def __init__(self, count, transform=None):
#         self.count = count
#         self.transform = transform

#     def __len__(self):
#         return self.count

#     def __getitem__(self, idx):
#         noise_img = np.random.normal(loc=0.5, scale=0.5, size=(28, 28)).clip(0, 1)
#         noise_img = Image.fromarray((noise_img * 255).astype(np.uint8), mode='L')
#         if self.transform:
#             noise_img = self.transform(noise_img)
#         return noise_img, 10  # Label 10 for noise

# # Prepare DataLoader for the test directory
# test_transform = transforms.Compose([
#     transforms.Resize((28, 28)),  # Ensure the images are the correct size
#     transforms.Grayscale(),       # Convert images to grayscale
#     transforms.ToTensor(),        # Convert images to tensor format
#     transforms.Normalize((0.5,), (0.5,))  # Normalize images
# ])
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])


# # Load MNIST data
# mnist_test = MNIST('./dataset', train=False, download=True, transform=test_transform)
# mnist_loader = DataLoader(mnist_test, batch_size=1, shuffle=False)

# # Save 5 images per MNIST digit
# saved_images = {i: 0 for i in range(10)}
# for img, label_tensor in mnist_loader:
#     label = label_tensor.item()  # Convert tensor to integer
#     if saved_images[label] < 5:
#         img_path = os.path.join(test_directory, str(label), f'label_{label}_img_{saved_images[label]}.png')
#         img = transforms.ToPILImage()(img.squeeze(0))  # Convert tensor to PIL Image and remove batch dimension
#         img.save(img_path)
#         saved_images[label] += 1
#     if all(count == 5 for count in saved_images.values()):
#         break

# # Save noise images
# noise_data = NoiseDataset(5, transform=test_transform)
# noise_loader = DataLoader(noise_data, batch_size=1, shuffle=False)
# for idx, (image, label) in enumerate(noise_loader):
#     label = label.item()  # Convert tensor to integer if it's not already
#     img_path = os.path.join(test_directory, str(label), f'label_{label}_img_{idx}.png')
#     pil_image = transforms.ToPILImage()(image.squeeze(0))  # Convert tensor to PIL Image and remove batch dimension if necessary
#     pil_image.save(img_path)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Classifier().to(device)
# model.eval()
# classifier_path = './classifier_ckpts/model.pt'
# ckpt = torch.load(classifier_path, map_location=device)
# model.load_state_dict(ckpt)


# test_dataset = ImageFolder('test', transform=test_transform)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# # Predict and print labels
# for images, _ in test_loader:
#     images = images.to(device)
#     with torch.no_grad():
#         outputs = model(images)
#         probs = outputs.exp()
#         for i in range(probs.shape[0]):
#             max_prob = probs[i].argmax()
#             print(f"max_prob: {max_prob}")

