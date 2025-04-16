import matplotlib.pyplot as plt
import numpy as np
import yaml
import argparse
import random
import torch
import shutil
import glob
import os

def plot_loss(train_losses, val_losses, save_path):
    """Plot and save the training and validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Training progress plot saved to {save_path}")

# Load configuration from a YAML file
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Parser for command-line arguments
def get_parser():
    parser = argparse.ArgumentParser(description="ConvTasNet Training Script with SI-SDR")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config file")
    return parser

def print_divider():
    terminal_width = shutil.get_terminal_size().columns
    print("=" * terminal_width)
    print(" ")


def set_seed(seed: int):
    random.seed(seed)
    
    np.random.seed(seed)

    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def normalize_tensor_min_max(tensor):
    max_val = torch.max(tensor)
    min_val = torch.min(tensor)
    
    # Normalize the tensor between -1 and 1
    normalized_tensor = 2 * (tensor - min_val) / (max_val - min_val) - 1
    return normalized_tensor

# Function to scan the 'models/' directory for available models
def find_trained_models():
    model_dirs = glob.glob('../../../results/DeepACE_V1_old/run_*')
    if not model_dirs:
        raise FileNotFoundError("No trained models found.")
    
    # Sort directories by modification time (latest first)
    model_dirs.sort(key=os.path.getmtime, reverse=False)
    
    # Find all model paths (.pth files) in each directory
    model_paths = []
    for model_dir in model_dirs:
        model_files = glob.glob(os.path.join(model_dir, '*.pth'))
        if model_files:
            model_paths.extend(model_files)
    
    if not model_paths:
        raise FileNotFoundError(f"No model files (.pth) found in 'models/'.")
    
    return model_paths