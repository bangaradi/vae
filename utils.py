import yaml
from datetime import datetime
import os
# import wandb
import argparse
from torchvision import datasets, transforms

def get_config_and_setup_dirs(filename):
    with open(filename, 'r') as fp:
        config = yaml.safe_load(fp)
    config = dict2namespace(config)
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    config.exp_root_dir = os.path.join("./results", config.dataset.lower(), timestamp)
    config.log_dir = os.path.join(config.exp_root_dir, 'logs')
    config.ckpt_dir = os.path.join(config.exp_root_dir, 'ckpts')
    os.makedirs(config.log_dir)
    os.makedirs(config.ckpt_dir)

    # wandb_id = wandb.util.generate_id()
    # config.wandb_id = wandb_id

    with open(os.path.join(config.exp_root_dir, 'config.yaml'), 'w') as fp:
        yaml.dump(config, fp)
    
    return config

def get_config_and_setup_dirs_final(working_dir):
    with open(working_dir, 'r') as fp:
        config = yaml.safe_load(fp)
    config = dict2namespace(config)
    
    # timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    config.exp_root_dir = os.path.join(f"./{config.working_dir}", config.dataset.lower(), "initial")
    config.log_dir = os.path.join(config.exp_root_dir, 'logs')
    config.ckpt_dir = os.path.join(config.exp_root_dir, 'ckpts')
    if not os.path.exists(config.exp_root_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)

    # wandb_id = wandb.util.generate_id()
    # config.wandb_id = wandb_id

    with open(os.path.join(config.exp_root_dir, f'config_initial.yaml'), 'w') as fp:
        yaml.dump(config, fp)
    
    return config

def setup_dirs_final(config, working_dir, filename):
    
    # timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    config.exp_root_dir = os.path.join(f"./{working_dir}", config.dataset.lower(), filename)
    config.log_dir = os.path.join(config.exp_root_dir, 'logs')
    config.ckpt_dir = os.path.join(config.exp_root_dir, 'ckpts')
    os.makedirs(config.log_dir)
    os.makedirs(config.ckpt_dir)

    # wandb_id = wandb.util.generate_id()
    # config.wandb_id = wandb_id

    with open(os.path.join(config.exp_root_dir, f'config_{filename}.yaml'), 'w') as fp:
        yaml.dump(config, fp)
    
    return config

def setup_dirs(config):
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    config.exp_root_dir = os.path.join("./results", config.dataset.lower(), timestamp)
    config.log_dir = os.path.join(config.exp_root_dir, 'logs')
    config.ckpt_dir = os.path.join(config.exp_root_dir, 'ckpts')
    os.makedirs(config.log_dir)
    os.makedirs(config.ckpt_dir)

    # wandb_id = wandb.util.generate_id()
    # config.wandb_id = wandb_id

    with open(os.path.join(config.exp_root_dir, 'config.yaml'), 'w') as fp:
        yaml.dump(config, fp)
    
    return config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def cycle(dl):
    while True:
        for data in dl:
            yield data
            
