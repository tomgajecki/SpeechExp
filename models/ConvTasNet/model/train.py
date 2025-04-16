import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import Dataset, collate_fn
from si_sdr_loss import si_sdr_loss
import argparse
import datetime 
import shutil 
import os
from utils import *
from model import ConvTasNet
from tqdm import tqdm

set_seed(42)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# Main training function
def main(config, config_path):
    # Create the dataset
    train_dataset = Dataset(
        config['train_mixture_dir'], config['train_target_dir'], 
        sample_rate=config['sample_rate'], segment_length=config['segment_length'])
    
    valid_dataset = Dataset(
        config['valid_mixture_dir'], config['valid_target_dir'], 
        sample_rate=config['sample_rate'], segment_length=config['segment_length'])

    # Create the DataLoader
    train_data_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, 
        num_workers=config['num_workers'], collate_fn=collate_fn)
    
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=config['batch_size'], shuffle=True, 
        num_workers=config['num_workers'], collate_fn=collate_fn)

    # Extract TasNet model parameters from the configuration
    tasnet_params = config['tasnet']

    # Initialize the ConvTasNet model using parameters from the config file
    model = ConvTasNet(
        L=tasnet_params['L'],
        N=tasnet_params['N'],
        P=tasnet_params['P'],
        B=tasnet_params['B'],
        S=tasnet_params['S'],
        H=tasnet_params['H'],
        R=tasnet_params['R'],
        X=tasnet_params['X'],
        msk_activate=tasnet_params['msk_activate'],
        causal=tasnet_params['causal']
    ).to(device)

    print("Model size: ", sum(p.numel() for p in model.parameters()))

    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=float(config['learning_rate']))

    # Define learning rate scheduler (ReduceLROnPlateau)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 25

    # Ensure the model_save_dir contains the timestamp
    model_save_dir = os.path.join('../../../results/ConvTasNet', f"run_{timestamp}")

    # Create the directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)

    # Copy the configuration file to the model_save_dir
    config_save_path = os.path.join(model_save_dir, 'config.yaml')
    shutil.copyfile(config_path, config_save_path)

    # Ensure the model save path includes the timestamped folder and file name
    model_save_path = os.path.join(model_save_dir, f"model_{timestamp}.pth")

    # Lists to track training and validation loss
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(config['num_epochs']):
        print_divider()
        
        model.train()
        total_loss = 0
        
        # Training progress bar
        train_loader_tqdm = tqdm(train_data_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}', leave=False)

        for _, (mix, target) in enumerate(train_loader_tqdm):
            mix = mix.to(device)
            target = target.to(device)

            # Forward pass: estimate the clean speech
            estimate = model(mix)

            # Calculate SI-SDR loss
            loss = si_sdr_loss(estimate, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Average loss for this epoch
        avg_loss = total_loss / len(train_data_loader)
        train_losses.append(avg_loss)  # Store training loss
        print(f'Epoch {epoch+1}:\nTraining Loss: {-avg_loss:.4f} dB')

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_loader_tqdm = tqdm(valid_data_loader, desc=f'Validation {epoch+1}/{config["num_epochs"]}', leave=False)
            for mix, target in val_loader_tqdm:
                mix = mix.to(device)
                target = target.to(device)

                estimate = model(mix)
                loss = si_sdr_loss(estimate, target)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(valid_data_loader)
            val_losses.append(avg_val_loss) 
            print(f'Validation Loss: {-avg_val_loss:.4f} dB')

            # Check if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                epochs_no_improve += 1
                print(f'Epochs without improvement: {epochs_no_improve}/{early_stop_patience}')
            
            # Early stopping
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered. No improvement in validation loss for {early_stop_patience} epochs.")
                break

            # Reduce learning rate on plateau and print the current learning rate
            scheduler.step(avg_val_loss)
            print(f'Learning rate: {scheduler.optimizer.param_groups[0]["lr"]}')
    
    # Save the plot of training and validation loss
    plot_loss(train_losses, val_losses, os.path.join(model_save_dir, 'training_progress.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ConvTasNet")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config file")
    args = parser.parse_args()

    # Load the configuration file
    config = load_config(args.config)

    # Check if GPU is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pass the configuration and its path to the main function
    main(config, args.config)