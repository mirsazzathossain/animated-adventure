import os
import torch
import yaml

config = {
        'model_params': {
            'name': 'LitResnet',
            'in_size': 32,
            'num_classes': 10,
        },
        'exp_params': {
            'lr': 0.05,
            'max_lr': 0.1,
            'accelerator': 'auto',
            'max_epochs': 30,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'batch_size': 256 if torch.cuda.is_available() else 64,
            'num_workers': int(os.cpu_count()),
            'devices': -1 if torch.cuda.is_available() else None,
            'path_dataset': 'data/cifar10',
        },
        'logging_params': {
            'save_dir': 'logs/lit_resnet',
            'name': 'LitResnet',
            'manual_seed': 7,
        },
    }

def write_config():
    """
    Writes the config file

    Returns:
        None
    """
    with open("configs/config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config():
    """
    Loads the config file

    Returns:
        config: Dictionary containing the config file.
    """
    config_file = open('./configs/config.yaml', 'r')
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    return config


if __name__ == '__main__':
    write_config()