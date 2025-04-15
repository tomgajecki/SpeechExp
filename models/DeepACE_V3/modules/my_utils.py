import matplotlib.pyplot as plt
import numpy as np
import yaml
import argparse
import random
import torch
import shutil
import glob
import os
import torch
import torch.nn as nn
import math
import matplotlib

def plot_loss(train_losses_dict, val_losses_dict, save_path):
    """
    Automatically align and plot training and validation losses, accounting for differences in epoch counts.

    Args:
        train_losses_dict (dict): Dictionary of training losses.
        val_losses_dict (dict): Dictionary of validation losses.
        save_path (str): Path to save the plot.
    """
    _, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # Epoch ranges for training and validation losses
    train_epochs = range(1, len(train_losses_dict["Rec. Loss"]) + 1)
    val_start_epoch = len(train_losses_dict["Rec. Loss"]) - len(val_losses_dict["Rec. Loss"]) + 1
    val_epochs = range(val_start_epoch, val_start_epoch + len(val_losses_dict["Rec. Loss"]))

    # Calculate warmup end epoch dynamically
    warmup_end_epoch = val_start_epoch

    # Plot Reconstruction Loss
    if "Rec. Loss" in train_losses_dict:
        axes[0].plot(train_epochs, train_losses_dict["Rec. Loss"], label="Train Reconstruction Loss")
    if "Rec. Loss" in val_losses_dict:
        axes[0].plot(val_epochs, val_losses_dict["Rec. Loss"], label="Val Reconstruction Loss", linestyle="--")
    
    # Add vertical line for warmup end
    axes[0].axvline(warmup_end_epoch, color='red', linestyle='--')
    axes[0].text(warmup_end_epoch, max(train_losses_dict["Rec. Loss"]), "Warmup End", 
                 color='red', rotation=90, verticalalignment='bottom', horizontalalignment='right')

    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Reconstruction Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Plot Discriminator Loss
    if "Disc. Loss" in train_losses_dict:
        axes[1].plot(train_epochs, train_losses_dict["Disc. Loss"], label="Train Discriminator Loss")
    if "Disc. Loss" in val_losses_dict:
        axes[1].plot(val_epochs, val_losses_dict["Disc. Loss"], label="Val Discriminator Loss", linestyle="--")

    # Add vertical line and label for warmup end in discriminator loss
    axes[1].axvline(warmup_end_epoch, color='red', linestyle='--')
    axes[1].text(warmup_end_epoch, max(train_losses_dict["Disc. Loss"]), "Warmup End", 
                 color='red', rotation=90, verticalalignment='bottom', horizontalalignment='right')

    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Discriminator Loss")
    axes[1].legend()
    axes[1].grid(True)

    # Finalize layout and save plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training and validation progress plot saved to {save_path}")


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
    print(" ")
    print("=" * terminal_width)
    print(" ")
    
def print_main_divider(title="Training Session Started"):
    terminal_width = shutil.get_terminal_size().columns
    top_border = "✦" * terminal_width
    bottom_border = "=" * terminal_width
    title_with_model = f"{title}"
    bold_title = f"\033[1m ✦ {title_with_model} ✦ \033[0m".center(terminal_width, " ")
    
    print("\n" + top_border + "\n")
    print(bold_title)
    print("\n" + bottom_border + "\n")


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
    model_dirs = glob.glob('../../../results/DeepACE_V3_office/run_*')
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



class AttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(AttentionPooling, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        
        # Shared linear layer for attention scores, output: [batch, seq_len, num_heads]
        self.attention = nn.Linear(input_dim, num_heads)
        
        # Optional projection layer to map back to input_dim after pooling
        self.projection = nn.Linear(input_dim, input_dim) if num_heads > 1 else None

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        
        # Compute attention scores for each head: [batch, seq_len, num_heads]
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)  # Normalize over sequence length
        
        # Compute weighted sum for each head: [batch, num_heads, input_dim]
        weighted_x = torch.einsum('bsl,bsn->bnl', x, weights)
        
        # Average or concatenate multi-head outputs
        if self.num_heads > 1:
            pooled = weighted_x.mean(dim=1)  # [batch, input_dim]
        else:
            pooled = weighted_x.squeeze(1)  # [batch, input_dim]

        # Optionally project back to input_dim
        if self.projection:
            pooled = self.projection(pooled)

        return pooled



def feature_normalize(x):
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)
    
def display_layer_gradients(model):
    print("Gradients for individual layers:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            mean_grad = param.grad.mean().item()
            std_grad = param.grad.std().item()
            print(f"Layer: {name} | Gradient Mean: {mean_grad:.6f} | Gradient Std: {std_grad:.6f}")
        else:
            print(f"Layer: {name} | No gradients available")

    print(f"Gradient Mean: {mean_grad:.6f}, Gradient Std: {std_grad:.6f}")


def gen_loss_plateaued(gen_losses, patience=5, threshold=1e-4):
    """Detects if generator loss has plateaued."""
    if len(gen_losses) < patience:
        return False
    recent_losses = gen_losses[-patience:]
    return max(recent_losses) - min(recent_losses) < threshold

def adjust_learning_rate(optimizer, base_lr, scale_factor):
    """Helper function to adjust learning rates for an optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * scale_factor


def compute_scale_factor(epoch, warmup_start, warmup_epochs):
    """Compute the scale factor for lambda and learning rate based on epoch."""
    if warmup_epochs == 1:
        return 1.0  # If there's only one warmup epoch, return the maximum scale immediately.

    warmup_epoch = epoch - warmup_start
    x = warmup_epoch / float(warmup_epochs - 1)  # Normalized epoch (0 to 1)
    return 1 / (1 + math.exp(-10 * (x - 0.5)))  # Sigmoid centered at 0.5

def update_lambdas(lambda_adv, lambda_feat, final_lambda_adv, final_lambda_feat, adv_loss, recon_loss, feature_matching_loss):
    target_lambda_adv = final_lambda_adv * (1.0 - adv_loss / (adv_loss + recon_loss + 1e-8))
    lambda_adv += 0.1 * (target_lambda_adv - lambda_adv)

    target_lambda_feat = final_lambda_feat * (
        adv_loss / (adv_loss + (feature_matching_loss if feature_matching_loss > 0 else 1e-8))
    )
    lambda_feat += 0.1 * (target_lambda_feat - lambda_feat)

    return lambda_adv, lambda_feat

import torch

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Computes the gradient penalty for WGAN-GP.

    Args:
        D (nn.Module): The discriminator (critic).
        real_samples (torch.Tensor): Real samples from the dataset.
        fake_samples (torch.Tensor): Fake samples generated by the generator.
        device (str): Device ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Gradient penalty.
    """
    batch_size, C, T = real_samples.shape  # Adjust based on input shape
    alpha = torch.rand(batch_size, 1, 1, device=device)  # Random interpolation
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = D(interpolates)  # Forward pass
    grad_outputs = torch.ones_like(d_interpolates, device=device)  # Tensor for gradients

    # Compute gradients of outputs w.r.t. interpolated inputs
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)  # Flatten gradients
    gradient_norm = gradients.norm(2, dim=1)  # L2 norm
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()  # Gradient penalty term

    return gradient_penalty



def debug_plot_one_electrodogram(electrodograms, index=0):
    """
    Plots a single electrodogram from a batch while in debugging mode.

    Args:
        electrodograms (Tensor or np.array): Shape (batch_size, time) or (batch_size, channels, time).
        index (int): Index of the batch element to plot.
    """
    matplotlib.use("TkAgg") 
    # Convert to NumPy if it's a Torch tensor
    if isinstance(electrodograms, torch.Tensor):
        electrodograms = electrodograms.detach().cpu().numpy()

    # Plot
    plt.figure(figsize=(8, 3))
    plt.plot(electrodograms[0,:,:], label=f"Sample {index}")
    plt.legend()
    plt.grid(True)
    plt.title("Debugging: Electrodogram Sample")
    plt.show(block=False)  # Non-blocking to avoid VSCode freezing