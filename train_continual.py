import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from dataset import MNIST_Custom
from torchvision.utils import save_image, make_grid
import pickle
from model import OneHotCVAE, loss_function
from utils import setup_dirs, cycle
import os
import argparse
import logging
import copy
import numpy as np
from tqdm import tqdm


def parse_args_and_ckpt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--ckpt_folder", type=str, required=True, help="Path to folder of original VAE"
    )

    parser.add_argument(
        "--data_path", type=str, default="./dataset", help="Path to MNIST dataset"
    )
    
    parser.add_argument(
        "--label_to_learn", type=int, default=0, help='Which MNIST class to drop'
    )

    parser.add_argument(
        "--labels_to_learn", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help='Which MNIST classes to learn'
    )
    
    parser.add_argument(
        "--lmbda", type=float, default = 100, help = "Lambda hyperparameter for EWC term in loss"
    )
    
    parser.add_argument(
        "--gamma", type=float, default = 1, help = "Gamma hyperparameter for contrastive term in loss (left at 1 in main paper)"
    )
    
    parser.add_argument(
        "--n_iters", type=int, default=50000, help="Number of iterations"
    )

    parser.add_argument(
        "--log_freq", type=int, default = 200, help='Logging frequency while training'
    )
    
    parser.add_argument(
        "--n_vis_samples", type=int, default=100, help='Number of samples to visualize while logging'
    )
    
    parser.add_argument(
        "--lr", type=float, default=1e-4, help='Learning rate'
    )
    
    parser.add_argument(
        "--batch_size", type=int, default=256, help='Batch size for training'
    )
    
    args = parser.parse_args()
    ckpt = torch.load(os.path.join(args.ckpt_folder, "ckpts/ckpt_modified.pt"), map_location=device)
    old_config = ckpt['config']
    new_config = setup_dirs(copy.deepcopy(old_config))
    
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(new_config.log_dir, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(logging.INFO)
    
    return args, ckpt, old_config, new_config


def train():
    
    vae_clone = copy.deepcopy(vae)
    vae_clone.eval()
    
    # label_choices = list(range(10))
    # label_choices.remove(args.label_to_drop)

    labels_to_remember = ckpt['labels']
    labels_to_learn = args.labels_to_learn
    
    vae.train()
    train_loss = 0
    learning_loss = 0
    ewc_loss = 0
    
    for step in tqdm(range(0, args.n_iters)):
        # data, label = next(train_iter)
        # label = F.one_hot(label, 10)
        # data = data.to(device)
        # label = label.to(device)
        
        c_remember = torch.from_numpy(np.random.choice(labels_to_remember, size=args.batch_size)).to(device)
        c_remember = F.one_hot(c_remember, 10)
        z_remember = torch.randn((args.batch_size, new_config.z_dim)).to(device)
        
        # c_forget = (torch.ones(args.batch_size, dtype=int) * args.label_to_drop).to(device)
        # c_forget = F.one_hot(c_forget, 10)
        # out_forget = torch.rand((args.batch_size, 1, 28, 28)).to(device)
        # c_new = torch.from_numpy(np.random.choice(labels_to_learn, size=args.batch_size)).to(device)
        # c_new = F.one_hot(c_new, 10)
        out_new, c_new = next(train_iter)
        c_new = F.one_hot(c_new, 10)
        out_new = out_new.to(device)
        c_new = c_new.to(device)

        with torch.no_grad():
            out_remember = vae_clone.decoder(z_remember, c_remember).view(-1, 1, 28, 28)

        optimizer.zero_grad()
                
        # # corrupting loss
        # recon_batch, mu, log_var = vae(out_forget, c_forget)
        # loss = loss_function(recon_batch, out_forget, mu, log_var)
        
        # learning loss
        recon_batch, mu, log_var = vae(out_new, c_new)
        loss = loss_function(recon_batch, out_new, mu, log_var)
        
        # contrastive loss
        recon_batch, mu, log_var = vae(out_remember, c_remember)
        loss += args.gamma * loss_function(recon_batch, out_remember, mu, log_var)
        
        learning_loss += loss / args.log_freq
        
        # for n, p in vae.named_parameters():
        #     _loss = fisher_dict[n].to(device) * (p - params_mle_dict[n].to(device)) ** 2
        #     loss += args.lmbda * _loss.sum()
        #     ewc_loss += args.lmbda * _loss.sum() / args.log_freq
        
        loss.backward()
        train_loss += loss.item() / args.log_freq
        optimizer.step()
        
        if (step+1) % args.log_freq == 0:
            logging.info('Train Step: {} ({:.0f}%)\t Avg Train Loss Per Batch: {:.6f}'.format(
                step, 100. * step / args.n_iters, train_loss))
            logging.info('Avg Continual Learning Loss Per Batch: {:.6f}\t Avg EWC Loss Per Batch\n: {:.6f}'.format(
                learning_loss, ewc_loss
            ))
            sample(step)
            train_loss = 0
            learning_loss = 0
            ewc_loss = 0


def sample(step):
    vae.eval()
    with torch.no_grad():
        z = torch.randn((args.n_vis_samples, new_config.z_dim)).to(device)
        # c = torch.repeat_interleave(torch.arange(10), args.n_vis_samples//10).to(device)
        # c = F.one_hot(c, 10)
        total_labels = args.labels_to_learn + ckpt['labels']
        c = torch.from_numpy(np.array([i for i in total_labels for _ in range(args.n_vis_samples//len(total_labels))])).to(device)
        c = F.one_hot(c, 10)

        # change shape of z if needed
        if c.shape[0] != z.shape[0]:
            z = z[:c.shape[0]]
        
        out = vae.decoder(z, c).view(-1, 1, 28, 28)
        
        grid = make_grid(out, nrow = args.n_vis_samples//len(total_labels))
        save_image(grid, os.path.join(new_config.log_dir, "step_" + str(step) + ".png"))


if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args, ckpt, old_config, new_config = parse_args_and_ckpt()
    logging.info(f"CVAE continual learning training.")
    logging.info(f"Current Digits: {ckpt['labels']}, New Digits to Learn : {args.labels_to_learn}, lambda: {args.lmbda}, gamma: {args.gamma}")

    # MNIST Dataset
    train_dataset = MNIST_Custom(digits=args.labels_to_learn, data_path=args.data_path, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = MNIST_Custom(digits=args.labels_to_learn, data_path=args.data_path, train=False, transform=transforms.ToTensor(), download=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    train_iter = cycle(train_loader)

    # build model
    vae = OneHotCVAE(x_dim=new_config.x_dim, h_dim1= new_config.h_dim1, h_dim2=new_config.h_dim2, z_dim=new_config.z_dim)
    vae = vae.to(device)

    vae.load_state_dict(ckpt['model'])
    vae.train()
    
    params_mle_dict = {}
    for name, param in vae.named_parameters():
        params_mle_dict[name] = param.data.clone()

    with open(os.path.join(old_config.exp_root_dir, 'fisher_dict.pkl'), 'rb') as f:
        fisher_dict = pickle.load(f)
    
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)

    train()
    torch.save({
            "model": vae.state_dict(),
            "config": new_config
        },
        os.path.join(new_config.ckpt_dir, "ckpt.pt"))