# prerequisites
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import Subset
import argparse
import os
import logging
from tqdm import tqdm
from dataset import MNIST_Custom
from utils import get_config_and_setup_dirs_final, cycle
from model import OneHotCVAE, loss_function

NUM_TRAIN_EPOCHS = {
    1: 20000,
    2: 40000,
    3: 50000,
    4: 50000,
    5: 60000,
    6: 65000,
    7: 70000,
    8: 100000,
    9: 100000,
    10: 100000,
}

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--config", type=str, default="mnist.yaml", help="Path to config file"
    )
    
    parser.add_argument(
        "--data_path", type=str, default="./dataset", help="Path to MNIST dataset"
    )
    
    parser.add_argument(
        "--batch_size", type=int, default=256, help='Batch size for training'
    )
    
    parser.add_argument(
        "--n_iters", type=int, default=100000, help='Number of training iterations'
    )
    
    parser.add_argument(
        "--log_freq", type=int, default = 5000, help='Logging frequency while training'
    )
    
    parser.add_argument(
        "--n_vis_samples", type=int, default=100, help='Number of samples to visualize while logging'
    )
    
    parser.add_argument(
        "--lr", type=float, default=0.0001, help='Learning rate'
    )

    args = parser.parse_args()
    config = get_config_and_setup_dirs_final(args.config)

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(config.log_dir, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(logging.INFO)
    
    return args, config

def train(vae, args, config, optimizer, n_iters, device, train_iter, test_loader):
    vae.train()
    
    train_loss = 0
    for step in tqdm(range(0, n_iters)):
        data, label = next(train_iter)
        label = F.one_hot(label, 10)
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data, label)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if (step+1) % args.log_freq == 0:
            logging.info('Train Step: {} ({:.0f}%)\t Avg Train Loss Per Batch: {:.6f}\t Avg Test Loss Per Batch: {:.6f}'.format(
                step, 100. * step / args.n_iters, train_loss / args.log_freq, test(vae, test_loader, device)))
            sample(vae, args, config, step)
            train_loss = 0


def test(vae, test_loader, device):
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, label in test_loader:
            label = F.one_hot(label, 10)
            data = data.to(device)
            label = label.to(device)
            recon, mu, log_var = vae(data, label)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader)
    return test_loss


def sample(vae, args, config, step, device):
    vae.eval()
    with torch.no_grad():
        z = torch.randn((args.n_vis_samples, config.z_dim)).to(device)
        c = torch.repeat_interleave(torch.arange(10), args.n_vis_samples//10).to(device)
        c = F.one_hot(c, 10)

        out = vae.decoder(z, c).view(-1, 1, 28, 28)
        
        grid = make_grid(out, nrow = args.n_vis_samples//10)
        save_image(grid, os.path.join(config.log_dir, "step_" + str(step) + ".png"))

# def filter_by_label(dataset, labels):
#     indices = [i for i, (img, label) in enumerate(dataset) if label in labels]
#     return Subset(dataset, indices)

def train_specialized_model(args, config, labels_to_learn):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Beginning basic training of specialized VAE")

    # MNIST Dataset
    train_dataset = MNIST_Custom(digits=labels_to_learn, data_path=args.data_path, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = MNIST_Custom(digits=labels_to_learn, data_path=args.data_path, train=True, transform=transforms.ToTensor(), download=True)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    train_iter = cycle(train_loader)

    # build model
    vae = OneHotCVAE(x_dim=config.x_dim, h_dim1= config.h_dim1, h_dim2=config.h_dim2, z_dim=config.z_dim)
    vae = vae.to(device)

    optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    
    train(vae, args, config, optimizer, NUM_TRAIN_EPOCHS[len(labels_to_learn)], device, train_iter, test_loader)
    torch.save({
            "model": vae.state_dict(),
            "config": config
        },
        os.path.join(config.ckpt_dir, "specialized_model.pt"))