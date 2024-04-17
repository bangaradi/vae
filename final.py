
# --config
# --data_path
# --batch_size
# --labels_to_learn_initial
# --n_iters_initial
# --n_iters_continual
# --n_iters_forget
# --log_freq
# --n_vis_samples
# --lr
# --lmbda
# --gamma
# --input_file
import gc
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms
import pickle
from model import OneHotCVAE, loss_function, SelectiveDropout
from utils import setup_dirs
import os
import argparse
import logging
import copy
import random
from dataset import MNIST_Custom
import numpy as np
from tqdm import tqdm
from utils import get_config_and_setup_dirs_final, cycle, find_indices_to_drop, prune_model, prune_model_using_dag, expand_model, evaluate_with_classifier

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
WARMUP_PERIOD = 10000
BREATHING_PERIOD = 5000
WARMED_UP = 0
HYPERPARAMETER_COMPRESS = 0.1
HYPERPARAMETERS_EXPAND = 0.1
# CLASSIFIER_PATH = './classifier_ckpts/model.pt'
CLASSIFIER_PATH = ["/home/stud-1/aditya/vae/classifier_ckpts/model1.pt", "/home/stud-1/aditya/vae/classifier_ckpts/model2.pt", "/home/stud-1/aditya/vae/classifier_ckpts/model3.pt","/home/stud-1/aditya/vae/classifier_ckpts/model4.pt", "/home/stud-1/aditya/vae/classifier_ckpts/model5.pt"]
METRIC_PATH = {
    'accuracy_values' : 'metrics/accuracy_values.csv',
    'acc_path' : 'metrics/acc.csv',
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
        "--labels_to_learn_initial", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help='Which MNIST classes to learn'
    )

    parser.add_argument(
        "--n_iters_initial", type=int, default=100000, help='Number of training iterations for initial model setup'
    )

    parser.add_argument(
        "--n_iters_continual", type=int, default=100000, help='Number of training iterations for learning new classes'
    )
    parser.add_argument(
        "--n_iters_forget", type=int, default=100000, help='Number of training iterations for forgettin classes'
    )
    
    parser.add_argument(
        "--log_freq", type=int, default = 5000, help='Logging frequency while training'
    )
    
    parser.add_argument(
        "--n_vis_samples", type=int, default=100, help='Number of samples to visualize while logging'
    )
    
    parser.add_argument(
        "--n_fim_samples", type=int, default=50, help='Number of samples to visualize while logging'
    )
    
    parser.add_argument(
        "--lr", type=float, default=0.0001, help='Learning rate'
    )

    parser.add_argument(
        "--lmbda", type=float, default=0.0001, help='Learning rate'
    )

    parser.add_argument(
        "--gamma", type=float, default=0.0001, help='Learning rate'
    )

    parser.add_argument(
        "--input_file", type=str, help='path to the input file'
    )
    
    parser.add_argument(
        "--user", type=str, default="sid", help='name of the user'
    )
    
    parser.add_argument(
        "--n_passes", type=int, default=10, help='number of learning and unlearning passes for the model'
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

def save_fim(vae, device, args, config):
    vae.train()
    fisher_dict = {}
    params_mle_dict = {}
    
    for name, param in vae.named_parameters():
        params_mle_dict[name] = param.data.clone()
        fisher_dict[name] = param.data.clone().zero_()
    
    for _ in tqdm(range(args.n_fim_samples)):
        
        with torch.no_grad():
            z = torch.randn(1, config.z_dim).to(device)
            c = torch.randint(0,10, (1,)).to(device)
            c = F.one_hot(c, 10)
            vae.eval()
            sample = vae.decoder(z, c)
        
        vae.train()
        vae.zero_grad()
        recon_batch, mu, log_var = vae(sample, c)
        loss = loss_function(recon_batch, sample, mu, log_var)
        loss.backward()

        for name, param in vae.named_parameters():
            if torch.isnan(param.grad.data).any():
                print("NAN detected")
            fisher_dict[name] += ((param.grad.data) ** 2) / args.n_fim_samples
        
    with open(os.path.join(config.exp_root_dir, 'fisher_dict.pkl'), 'wb') as f:
        pickle.dump(fisher_dict, f)

def train_initial(LEARNT_LABELS, labels_to_learn, optimizer_name, n_iter, device, args, config, line_count):
    LEARNT_LABELS.extend(labels_to_learn)
    print("learnt labels : ", LEARNT_LABELS)
    train_dataset = MNIST_Custom(digits=labels_to_learn, data_path=args.data_path, train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    train_iter = cycle(train_loader)
    vae = OneHotCVAE(x_dim=config.x_dim, h_dim1= config.h_dim1, h_dim2=config.h_dim2, z_dim=config.z_dim)
    vae = vae.to(device)

    if optimizer_name == 'adam':
        optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    
    vae.train()
    
    train_loss = 0
    for step in tqdm(range(0, n_iter)):
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
                step, 100. * step / n_iter, train_loss / args.log_freq, test(labels_to_learn, vae, device, args)))
            sample(step, vae, device, args, config, "initial", line_count)
            train_loss = 0
    
    
    # save the model
    state_dict_to_save = vae.state_dict()
    torch.save({
            "model": state_dict_to_save,
            "config": config,
            "labels": labels_to_learn,
            "fisher_dict": None, # TODO: save fisher dict
            "params_mle_dict": None, # TODO: save params_mle_dict
            "h_dims1": state_dict_to_save['fc1.weight'].shape[0], #TODO:
            "h_dims2": state_dict_to_save['fc2.weight'].shape[0], #TODO:
        },
        os.path.join(config.ckpt_dir, "ckpt_modified.pt"))
    
    cum_sum, avg_entropy = evaluate_with_classifier(config.ckpt_dir, CLASSIFIER_PATH, LEARNT_LABELS, [], METRIC_PATH, config)
    save_fim(vae, device, args, config)

    return LEARNT_LABELS

def train_continual(labels_to_learn, optimizer_name, n_iter, vae, device, args, config, line_count):
    global WARMED_UP, BREATHING_PERIOD, WARMUP_PERIOD, HYPERPARAMETERS_EXPAND, LEARNT_LABELS
    # take union of learnt labels and new labels
    labels_to_remember = LEARNT_LABELS.copy()
    size_before = len(LEARNT_LABELS)
    LEARNT_LABELS = list(set(LEARNT_LABELS).union(set(labels_to_learn)))
    print("learnt labels : ", LEARNT_LABELS)
    size_now = len(LEARNT_LABELS)

    train_dataset = MNIST_Custom(digits=labels_to_learn, data_path=args.data_path, train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    train_iter = cycle(train_loader)

    vae_clone = copy.deepcopy(vae)
    vae_clone.eval()
    
    # expand the model :
    print("EXPANDING MODEL")
    h_dim1 =int(vae.state_dict()['fc1.weight'].shape[0] * HYPERPARAMETERS_EXPAND) + vae.state_dict()['fc1.weight'].shape[0]
    h_dim2 = int(vae.state_dict()['fc2.weight'].shape[0] * HYPERPARAMETERS_EXPAND) + vae.state_dict()['fc2.weight'].shape[0]
    
    print("before expanding the model")
    for layer_name in vae.state_dict():
        print(f"Layer {layer_name} shape: {vae.state_dict()[layer_name].shape}")
        
    state_dict_expanded = expand_model(vae.state_dict(), HYPERPARAMETERS_EXPAND)
    vae = OneHotCVAE(x_dim=config.x_dim, h_dim1=h_dim1, h_dim2=h_dim2, z_dim=config.z_dim)
    vae.load_state_dict(state_dict_expanded)
    vae = vae.to(device)
    if optimizer_name == 'adam':
        optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    vae.train()
    train_loss = 0
    learning_loss = 0
    ewc_loss = 0
    for step in tqdm(range(0, n_iter)):
        if step >= WARMUP_PERIOD:
            WARMED_UP = 1
        c_remember = torch.from_numpy(np.random.choice(labels_to_remember, size=args.batch_size)).to(device)
        c_remember = F.one_hot(c_remember, 10)
        z_remember = torch.randn((args.batch_size, config.z_dim)).to(device)

        out_new, c_new = next(train_iter)
        c_new = F.one_hot(c_new, 10)
        out_new = out_new.to(device)
        c_new = c_new.to(device)

        with torch.no_grad():
            out_remember = vae_clone.decoder(z_remember, c_remember)
        
        optimizer.zero_grad()

        #learning_loss
        recon_batch, mu, log_var = vae(out_remember, c_remember)
        loss = loss_function(recon_batch, out_remember, mu, log_var)

        #contrastive loss
        recon_batch, mu, log_var = vae(out_new, c_new)
        loss += loss_function(recon_batch, out_new, mu, log_var)

        learning_loss += loss / args.log_freq

        # for n, p in vae.named_parameters():
        #     _loss = fisher_dict[n].to(device) * (p - params_mle_dict[n].to(device)) ** 2
        #     loss += args.lmbda * _loss.sum()
        #     ewc_loss += args.lmbda * _loss.sum() / args.log_freq
        loss.backward()
        train_loss += loss.item() / args.log_freq
        optimizer.step()

        # ADDING SELECTIVE DROPOUT
        if WARMED_UP and step % BREATHING_PERIOD == 0:
            # print("Warned up and ready")
            for layer_name, _ in vae.named_parameters():
                if 'fc' in layer_name and 'weight' in layer_name:
                    attribute_name = layer_name.split('.')[0]
                    base_layer_name = layer_name.rsplit('.', 1)[0]
                    indices = find_indices_to_drop(vae.state_dict(), base_layer_name)
                    current_layer = getattr(vae, attribute_name)
                    
                    if isinstance(current_layer, nn.Sequential) and any(isinstance(layer, SelectiveDropout) for layer in current_layer):
                        # If the layer is already a nn.Sequential with a SelectiveDropout, update the dropout layer
                        for i, layer in enumerate(current_layer):
                            if isinstance(layer, SelectiveDropout):
                                current_layer[i] = SelectiveDropout(dropout_rate=0.5, neuron_indices=indices)
                    else:
                        # If it's not modified yet, add the dropout layer
                        dropout_layer = SelectiveDropout(dropout_rate=0.5, neuron_indices=indices)
                        modified_layer = nn.Sequential(current_layer, dropout_layer)
                        setattr(vae, attribute_name, modified_layer)

        if (step+1) % args.log_freq == 0:
            logging.info('Train Step: {} ({:.0f}%)\t Avg Train Loss Per Batch: {:.6f}'.format(
                step, 100. * step / n_iter, train_loss))
            logging.info('Avg Continual Learning Loss Per Batch: {:.6f}\t Avg EWC Loss Per Batch\n: {:.6f}'.format(
                learning_loss, ewc_loss
            ))
            logging.info('Avg Test Loss Per Batch: {:.6f}'.format(test(LEARNT_LABELS, vae, device, args)))
            sample(step, vae, device, args, config, "continual", line_count)
            train_loss = 0
            learning_loss = 0
            ewc_loss = 0
    
    # save the model
    # state_dict_to_save = prune_model_using_dag(vae.state_dict())
    model_state_dict = vae.state_dict()
    for key in list(model_state_dict.keys()):
        if '.0.' in key:
            new_key = key.replace('.0.', '.')
            model_state_dict[new_key] = model_state_dict.pop(key)
    state_dict_to_save = model_state_dict
    # print("state_dict_to_save : ", state_dict_to_save.keys())
    torch.save({
            "model": state_dict_to_save,
            "config": config,
            "labels": LEARNT_LABELS,
            "fisher_dict": None, # TODO: save fisher dict
            "params_mle_dict": None, # TODO: save params_mle_dict
            "h_dims1": state_dict_to_save['fc1.weight'].shape[0], #TODO:
            "h_dims2": state_dict_to_save['fc2.weight'].shape[0], #TODO:
        },
        os.path.join(config.ckpt_dir, "ckpt_modified.pt"))

    cum_sum, avg_entropy = evaluate_with_classifier(config.ckpt_dir, CLASSIFIER_PATH, LEARNT_LABELS, [], METRIC_PATH, config)
    save_fim(vae, device, args, config)
    WARMED_UP = 0
    return LEARNT_LABELS

def train_forget(labels_to_forget, optimizer_name, n_iter, vae, device, args, config, line_count):
    global WARMED_UP, BREATHING_PERIOD, WARMUP_PERIOD, LEARNT_LABELS
    vae_clone = copy.deepcopy(vae)
    vae_clone.eval()

    LEARNT_LABELS = [i for i in LEARNT_LABELS if i not in labels_to_forget]
    print("learnt labels : ", LEARNT_LABELS)

    # train_dataset = MNIST_Custom(digits=LEARNT_LABELS, data_path=args.data_path, train=True, transform=transforms.ToTensor(), download=True)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # train_iter = cycle(train_loader)

    params_mle_dict = {}
    for name, param in vae.named_parameters():
        params_mle_dict[name] = param.data.clone()
    
    with open(os.path.join(config.exp_root_dir, 'fisher_dict.pkl'), 'rb') as f:
        fisher_dict = pickle.load(f)

    print("before compression:")
    for layer_name in vae.state_dict():
        print(f"Layer {layer_name} shape: {vae.state_dict()[layer_name].shape}")

    state_dict_pruned = prune_model_using_dag(vae.state_dict(), type=2)
    vae = OneHotCVAE(x_dim=config.x_dim, h_dim1=state_dict_pruned['fc1.weight'].shape[0], h_dim2=state_dict_pruned['fc2.weight'].shape[0], z_dim=config.z_dim)
    vae.load_state_dict(state_dict_pruned)
    vae = vae.to(device)
    if optimizer_name == 'adam':
        optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    vae.train()
    train_loss = 0
    forgetting_loss = 0
    ewc_loss = 0

    for step in tqdm(range(0, n_iter)):
        if step >= WARMUP_PERIOD:
            WARMED_UP = 1
        c_remember = torch.from_numpy(np.random.choice(LEARNT_LABELS, size=args.batch_size)).to(device)
        c_remember = F.one_hot(c_remember, 10)
        z_remember = torch.randn((args.batch_size, config.z_dim)).to(device)

        c_forget = torch.from_numpy(np.random.choice(labels_to_forget, size=args.batch_size)).to(device)
        c_forget = F.one_hot(c_forget, 10)
        out_forget = torch.rand((args.batch_size, 1, 28, 28)).to(device)

        with torch.no_grad():
            out_remember = vae_clone.decoder(z_remember, c_remember).view(-1, 1, 28, 28)
        
        optimizer.zero_grad()

        #corrupting loss
        recon_batch, mu, log_var = vae(out_forget, c_forget)
        loss = loss_function(recon_batch, out_forget, mu, log_var)

        #contrastive loss
        recon_batch, mu, log_var = vae(out_remember, c_remember)
        loss += args.gamma * loss_function(recon_batch, out_remember, mu, log_var)

        forgetting_loss += loss / args.log_freq

        # for n, p in vae.named_parameters():
        #     _loss = fisher_dict[n].to(device) * (p - params_mle_dict[n].to(device)) ** 2
        #     loss += args.lmbda * _loss.sum()
        #     ewc_loss += args.lmbda * _loss.sum() / args.log_freq
        
        loss.backward()
        train_loss += loss.item() / args.log_freq
        optimizer.step()

        # ADDING SELECTIVE DROPOUT
        if WARMED_UP and step % BREATHING_PERIOD == 0:
            # print("Warned up and ready")
            for layer_name, _ in vae.named_parameters():
                if 'fc' in layer_name and 'weight' in layer_name:
                    attribute_name = layer_name.split('.')[0]
                    base_layer_name = layer_name.rsplit('.', 1)[0]
                    indices = find_indices_to_drop(vae.state_dict(), base_layer_name)
                    current_layer = getattr(vae, attribute_name)
                    
                    if isinstance(current_layer, nn.Sequential) and any(isinstance(layer, SelectiveDropout) for layer in current_layer):
                        # If the layer is already a nn.Sequential with a SelectiveDropout, update the dropout layer
                        for i, layer in enumerate(current_layer):
                            if isinstance(layer, SelectiveDropout):
                                current_layer[i] = SelectiveDropout(dropout_rate=0.5, neuron_indices=indices)
                    else:
                        # If it's not modified yet, add the dropout layer
                        dropout_layer = SelectiveDropout(dropout_rate=0.5, neuron_indices=indices)
                        modified_layer = nn.Sequential(current_layer, dropout_layer)
                        setattr(vae, attribute_name, modified_layer)

        if (step+1) % args.log_freq == 0:
            logging.info('Train Step: {} ({:.0f}%)\t Avg Train Loss Per Batch: {:.6f}'.format(
                step, 100. * step / n_iter, train_loss))
            logging.info('Avg Forgetting Loss Per Batch: {:.6f}\t Avg EWC Loss Per Batch\n: {:.6f}'.format(
                forgetting_loss, ewc_loss
            ))
            logging.info('Avg Test Loss Per Batch: {:.6f}'.format(test(LEARNT_LABELS, vae, device, args)))
            sample(step, vae, device, args, config, "forget", line_count)
            train_loss = 0
            forgetting_loss = 0
            ewc_loss = 0
    
    # state_dict_to_save = prune_model_using_dag(vae.state_dict())
    state_dict_to_save = vae.state_dict()
    # remove .0. from the keys
    for key in list(state_dict_to_save.keys()):
        if '.0.' in key:
            new_key = key.replace('.0.', '.')
            state_dict_to_save[new_key] = state_dict_to_save.pop(key)
    #save the model
    torch.save({
            "model": state_dict_to_save,
            "config": config,
            "labels": LEARNT_LABELS,
            "fisher_dict": fisher_dict, # TODO: save fisher dict
            "params_mle_dict": params_mle_dict, # TODO: save params_mle_dict
            "h_dims1": state_dict_to_save['fc1.weight'].shape[0], #TODO
            "h_dims2": state_dict_to_save['fc2.weight'].shape[0], #TODO
        },
        os.path.join(config.ckpt_dir, "ckpt_modified.pt"))

    cum_sum, avg_entropy = evaluate_with_classifier(config.ckpt_dir, CLASSIFIER_PATH, LEARNT_LABELS, labels_to_forget, METRIC_PATH, config)
    save_fim(vae, device, args, config)

    return LEARNT_LABELS

def test(labels_to_learn, vae, device, args):
    test_dataset = MNIST_Custom(digits=labels_to_learn, data_path=args.data_path, train=False, transform=transforms.ToTensor(), download=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

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
    

def sample(step, vae, device, args, config, folder_name, line_count):
    global LEARNT_LABELS
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    vae.eval()
    # with torch.no_grad():
    #     # z = torch.randn((args.n_vis_samples, config.z_dim)).to(device)
    #     # c = torch.repeat_interleave(torch.arange(10), args.n_vis_samples//10).to(device)
    #     # c = F.one_hot(c, 10)
        
    #     z = torch.randn((args.n_vis_samples, config.z_dim)).to(device)
    #     # c = torch.from_numpy(np.random.choice(args.labels_to_learn_initial, size=args.n_vis_samples)).to(device) 
    #     # one digit in one row
    #     c = torch.from_numpy(np.array([i for i in args.labels_to_learn_initial for _ in range(args.n_vis_samples//len(args.labels_to_learn_initial))])).to(device)
    #     c = F.one_hot(c, 10)

    #     out = vae.decoder(z, c).view(-1, 1, 28, 28)
        
    #     grid = make_grid(out, nrow = args.n_vis_samples//len(args.labels_to_learn_initial))
    #     print("inside sample step : ", config.log_dir, folder_name + str(line_count), "step_" + str(step) + ".png")
    #     # make if the directory doesn't exist
    #     if not os.path.exists(os.path.join(config.log_dir, folder_name + str(line_count))):
    #         os.makedirs(os.path.join(config.log_dir, folder_name + str(line_count)))
    #     save_image(grid, os.path.join(config.log_dir, folder_name + str(line_count), "step_" + str(step) + ".png"))
    with torch.no_grad():  # Disable gradient tracking
        # Total number of images to generate
        images_per_label = 10
        total_images = len(digits) * images_per_label

        # Generate a batch of latent vectors
        z = torch.randn((total_images, config.z_dim)).to(device)

        # Generate labels for LEARNT_LABELS, repeating each label 10 times
        labels_to_use = np.repeat(digits, images_per_label)
        c = torch.tensor(labels_to_use, dtype=torch.long).to(device)
        c = F.one_hot(c, num_classes=10)  # Adjust num_classes as needed for your model

        # Decode the batch to get generated images
        out = vae.decoder(z, c).view(-1, 1, 28, 28)  # Assuming the images are 28x28 pixels

        # Create a grid of images with 10 images per row
        grid = make_grid(out, nrow=images_per_label)

        # Ensure the directory exists
        save_path = os.path.join(config.log_dir, folder_name + str(line_count))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save the grid to a PNG file
        save_image(grid, os.path.join(save_path, f"step_{step}.png"))


def main():
    global LEARNT_LABELS
    LEARNT_LABELS = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args, config = parse_args_and_config()
    # print(args)
    # logging.info(f"Beginning basic training of conditional VAE")

    
    # # build model
    # vae = OneHotCVAE(x_dim=config.x_dim, h_dim1= config.h_dim1, h_dim2=config.h_dim2, z_dim=config.z_dim)
    # vae = vae.to(device)

    # optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    
    # train_initial()
    # torch.save({
    #         "model": vae.state_dict(),
    #         "config": config,
    #         "labels": args.labels_to_learn_initial
    #     },
    #     os.path.join(config.ckpt_dir, "ckpt_modified.pt"))
    
    # after initial training is done, we put the training according to the input_file
    with open(args.input_file, 'r') as file:
        # get the number of lines in the file
        lines = file.readlines()
        # n_lines = len(lines)

    # initial training
    initial_labels = random.sample(range(10), random.randint(1, 10))
    optimizer_name = 'adam'
    n_iter = NUM_TRAIN_EPOCHS[len(initial_labels)]
    n_passes_completed = 0
    LEARNT_LABELS = train_initial(LEARNT_LABELS, initial_labels, optimizer_name, n_iter, device, args, config, n_passes_completed)

    n_passes = args.n_passes
    
    while n_passes_completed < n_passes:
        print(f"currently on pass number  {n_passes_completed}")
        action = random.randint(0, 1)  # 0 for forget, 1 for learn
        
        if action == 0:
            # if LEARNED_LABELS is empty, we can't forget anything, so go to the start of the loop again
            if len(LEARNT_LABELS) == 0:
                continue
            # sample (without replacement) a random number of labels to forget from the learnt labels
            labels_to_forget = random.sample(LEARNT_LABELS, random.randint(1, len(LEARNT_LABELS)))
            if len(labels_to_forget) == len(LEARNT_LABELS):
                continue # cannot unlearn entire model
            print(f"forgetting the labels {labels_to_forget}")
            n_iter = NUM_TRAIN_EPOCHS[len(labels_to_forget)]
        else:
            if len(LEARNT_LABELS) == 10:
                continue
            # sample (without replacement) a random set of labels to learn - and no label should belong to LEARNED_LABELS
            labels_to_learn = random.sample([i for i in range(10) if i not in LEARNT_LABELS], random.randint(1, 10 - len(LEARNT_LABELS)))
            print(f"learning the labels {labels_to_learn}")
            n_iter = NUM_TRAIN_EPOCHS[len(labels_to_learn)]
        
        optimizer_name = 'adam'
        # load the vae
        checkpoint = torch.load(os.path.join(config.ckpt_dir, "ckpt_modified.pt"))
        h_dim1 = checkpoint['h_dims1']
        h_dim2 = checkpoint['h_dims2']
        vae = OneHotCVAE(x_dim=config.x_dim, h_dim1=h_dim1, h_dim2=h_dim2, z_dim=config.z_dim)
        vae.load_state_dict(checkpoint['model'])
        vae = vae.to(device)
        
        if action == 0:
            LEARNT_LABELS = train_forget(labels_to_forget, optimizer_name, n_iter, vae, device, args, config, n_passes_completed)
        else:
            LEARNT_LABELS = train_continual(labels_to_learn, optimizer_name, n_iter, vae, device, args, config, n_passes_completed)
        
        n_passes_completed += 1

    
    # # first line : initial training details 
    # line_count = 0
    # line = lines[line_count]
    # line = line.strip().split()
    # labels_to_learn = [int(i) for i in line[1].split(',')]
    # optimizer_name = 'adam'
    # n_iter = int(line[2])
    # LEARNT_LABELS = train_initial(LEARNT_LABELS, labels_to_learn, optimizer_name, n_iter, device, args, config, line_count)

    # while True:
    #     line_count += 1
    #     # if line_count >= n_lines:
    #     #     break
    #     line = lines[line_count]
    #     line = line.strip().split()
    #     if line[0] == 'learn':
    #         labels_to_learn = [int(i) for i in line[1].split(',')]
    #         optimizer_name = 'adam'
    #         n_iter = int(line[2])
    #         # load the vae
    #         checkpoint = torch.load(os.path.join(config.ckpt_dir, "ckpt_modified.pt"))
    #         h_dim1 = checkpoint['h_dims1']
    #         h_dim2 = checkpoint['h_dims2']
    #         # print("h_dim1 : ", h_dim1, "h_dim2 : ", h_dim2)
    #         vae = OneHotCVAE(x_dim=config.x_dim, h_dim1=h_dim1, h_dim2=h_dim2, z_dim=config.z_dim)
    #         vae.load_state_dict(checkpoint['model'])
    #         vae = vae.to(device)
    #         LEARNT_LABELS = train_continual(labels_to_learn, optimizer_name, n_iter, vae, device, args, config, line_count)
    #     elif line[0] == 'forget':
    #         labels_to_forget = [int(i) for i in line[1].split(',')]
    #         optimizer_name = 'adam'
    #         n_iter = int(line[2])
    #         # load the vae
    #         checkpoint = torch.load(os.path.join(config.ckpt_dir, "ckpt_modified.pt"))
    #         h_dim1 = checkpoint['h_dims1']
    #         h_dim2 = checkpoint['h_dims2']
    #         # print("h_dim1 : ", h_dim1, "h_dim2 : ", h_dim2)
    #         vae = OneHotCVAE(x_dim=config.x_dim, h_dim1=h_dim1, h_dim2=h_dim2, z_dim=config.z_dim)
    #         vae.load_state_dict(checkpoint['model'])
    #         vae = vae.to(device)
    #         LEARNT_LABELS = train_forget(labels_to_forget, optimizer_name, n_iter, vae, device, args, config, line_count)
    #     else:
    #         continue

main()
print("Done") # (TODO: remove this line)