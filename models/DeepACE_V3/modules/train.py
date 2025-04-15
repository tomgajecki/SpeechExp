import math
import torch
import argparse
import datetime
import os
import numpy as np
import pandas as pd
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import Dataset, collate_fn
from generator import DeepACE
from discriminatorCNN import Discriminator
from losses import generator_loss, discriminator_loss
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(42)

def main(config, config_path, timestamp):
    # ---------------------
    #  Create the Dataset
    # ---------------------
    train_dataset = Dataset(
        config['train_mixture_dir'],
        config['train_target_dir'],
        sample_rate=config['sample_rate'],
        stim_rate=config['stim_rate'],
        segment_length=config['segment_length']
    )
    
    valid_dataset = Dataset(
        config['valid_mixture_dir'],
        config['valid_target_dir'],
        sample_rate=config['sample_rate'],
        stim_rate=config['stim_rate'],
        segment_length=config['segment_length']
    )

    # ---------------------
    #  Dataloader
    # ---------------------
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        drop_last=True
    )
    
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        drop_last=True
    )

    # ---------------------
    #  Initialize Models
    # ---------------------
    generator_params = config['DeepACE']
    discriminator_params = config['Discriminator']

    generator = DeepACE(
        L=generator_params['L'],
        N=generator_params['N'],
        P=generator_params['P'],
        B=generator_params['B'],
        S=generator_params['S'],
        H=generator_params['H'],
        R=generator_params['R'],
        X=generator_params['X'],
        M=generator_params['M'],
        msk_activate=generator_params['msk_activate'],
        causal=generator_params['causal'],
        base_level = config['base_level']
    ).to(device)

    discriminator = Discriminator(
        input_size=discriminator_params['input_size'],
        hidden_size=discriminator_params['hidden_size'],
        num_layers=discriminator_params['num_layers'],
        dropout=discriminator_params['dropout'],
        num_scales=discriminator_params['num_scales'],
        downsampling_factor=discriminator_params['downsampling_factor']

    ).to(device)

    generator_size = sum(p.numel() for p in generator.parameters())
    discriminator_size = sum(p.numel() for p in discriminator.parameters())

    print(f"\nModel Parameter Counts:")
    print(f"  Generator: {generator_size:,} parameters")
    print(f"  Discriminator: {discriminator_size:,} parameters")

    print_main_divider()
    
    # ---------------------
    #  Optimizers
    # ---------------------
    generator_optimizer = Adam(
        generator.parameters(),
        lr=float(config['gen_learning_rate']),
        betas=(0.5, 0.999)
    )
    discriminator_optimizer = Adam(
        discriminator.parameters(),
        lr=float(config['dis_learning_rate']),
        betas=(0.5, 0.99)
    )

    # ---------------------
    #  Schedulers
    # ---------------------
    ReduceLROnPlateau_patience = config['gen_reduce_lr_patience']
    generator_scheduler = ReduceLROnPlateau(
        generator_optimizer,
        mode='min',
        factor=0.5,
        patience=ReduceLROnPlateau_patience,
        min_lr=1e-6
    )
     
    ReduceLROnPlateau_patience = config['dis_reduce_lr_patience']
    discriminator_scheduler = ReduceLROnPlateau(
        discriminator_optimizer,
        mode='min',
        factor=0.5,
        patience=ReduceLROnPlateau_patience,
        min_lr=1e-6
    )

    # ---------------------
    #  Early Stopping Setup
    # ---------------------
    best_val_rec_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = config['early_stop_patience']

    # ---------------------
    #  Saving Paths
    # ---------------------
    # Ensure the model_save_dir contains the timestamp
    model_save_dir = os.path.join('../../../results/DeepACE_V3_office', f"run_{timestamp}")

    # Create the directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)

    # Copy the configuration file to the model_save_dir
    config_save_path = os.path.join(model_save_dir, 'config.yaml')
    shutil.copyfile(config_path, config_save_path)

    # Ensure the model save path includes the timestamped folder and file name
    model_save_path = os.path.join(model_save_dir, f"model_{timestamp}.pth")

    # ---------------------
    #  Setup for Logging
    # ---------------------
    train_losses = {
        'Rec. Loss': [],            # Reconstruction loss (primary metric for generator)
        'Disc. Loss': [],           # Total discriminator loss
    }

    val_losses = {
        'Rec. Loss': [],            # Validation reconstruction loss
        'Disc. Loss': [],           # Validation total discriminator loss
    }

    # ---------------------
    #  Loss Weighting
    # ---------------------
    final_lambda_adv = config['LossWeights']['lambda_adv']
    final_lambda_feat = config['LossWeights']['lambda_feat']
    warmup_epochs   = config['warmup_epochs']
    warmup_start = config['warmup_start']
    lambda_rec      = config['LossWeights']['lambda_rec']

    # ---------------------
    #  Learning Rate Warmup
    # ---------------------
    initial_dis_lr = float(config['dis_learning_rate'])
  
    warmup_complete = warmup_start == 0 and warmup_epochs == 0

    for epoch in range(config["num_epochs"]):
     
        if epoch < warmup_start:
            # Pre-Warmup Phase
            lambda_adv, lambda_feat = 0, 0
            if config['LossWeights']['include_feature_matching']:
                print(f"lambda_adv={lambda_adv:.4e}, lambda_feat={lambda_feat:.4e}")
            else:
                print(f"lambda_adv={lambda_adv:.4e}")

        elif warmup_start <= epoch < warmup_start + warmup_epochs:
            # Warmup Phase
            scale_factor = compute_scale_factor(epoch, warmup_start, warmup_epochs)
            lambda_adv = scale_factor * final_lambda_adv
            lambda_feat = scale_factor * final_lambda_feat if config['LossWeights']['include_feature_matching'] else 0.0

            # Cosine schedule for learning rate
            alpha = math.pi * scale_factor  # alpha in [0, π]
            lr_scale = 0.5 * (1.0 - math.cos(alpha))  # goes from 0 to 1

            adjust_learning_rate(discriminator_optimizer, initial_dis_lr, lr_scale)
            
            if warmup_start + warmup_epochs - 1 == epoch:
                warmup_complete = True
                print("Warmup complete!\n")
            
            if config['LossWeights']['include_feature_matching']:
                print(f"lambda_adv={lambda_adv:.4e}, lambda_feat={lambda_feat:.4e}")
            else:
                print(f"lambda_adv={lambda_adv:.4e}")
            
        else:

            lambda_adv = final_lambda_adv
            lambda_feat = final_lambda_feat

            adjust_learning_rate(discriminator_optimizer, initial_dis_lr, 1.0)

            if config['LossWeights']['include_feature_matching']:
                print(f"lambda_adv={lambda_adv:.4e}, lambda_feat={lambda_feat:.4e}")
            else:
                print(f"lambda_adv={lambda_adv:.4e}")

    
     
        # ---------------------
        #  Training Mode
        # ---------------------
        generator.train()
        discriminator.train()
        
        train_gen_losses = []
        train_disc_losses = []
        train_fake_losses = []
        train_real_losses = []
        train_rec_losses = []
        train_adv_losses = []
        if config['LossWeights']['include_feature_matching']:
            train_match_losses = []
        
        gen_iterations = 0

        # ---------------------
        #  Training Loop
        # ---------------------
        train_loader_tqdm = tqdm(
            train_data_loader,
            desc=f'Epoch {epoch+1}/{config["num_epochs"]}',
            leave=False,
            dynamic_ncols=True
        )
        train_iter = 0
        val_iter = 0
        for _, (noisy_audio, real_electrodograms) in enumerate(train_loader_tqdm):
            #if train_iter >= 20:
            #    break
            noisy_audio = noisy_audio.to(device)
            real_electrodograms = real_electrodograms.to(device)
            
            # 1) Train Generator
            generator_optimizer.zero_grad()
            generated_electrodograms = generator(noisy_audio)
            
            if config['LossWeights']['include_feature_matching']:
                
                real_outputs, real_features = discriminator(
                    real_electrodograms,
                    return_features=True
                )
                
                fake_outputs, fake_features = discriminator(
                    generated_electrodograms,
                    return_features=True
                )

                feature_matching_loss = 0
                for fake_feat, real_feat in zip(fake_features, real_features):
                    feature_matching_loss += torch.nn.MSELoss()(fake_feat, real_feat.detach())
                feature_matching_loss /= len(fake_features)

                gen_loss, rec_loss, adv_loss = generator_loss(
                    fake_outputs,
                    generated_electrodograms,
                    real_electrodograms,
                    lambda_adv,
                    lambda_rec
                )
                gen_loss = lambda_feat * feature_matching_loss + gen_loss

            else:
                fake_outputs = discriminator(generated_electrodograms)
                gen_loss, rec_loss, adv_loss = generator_loss(
                    fake_outputs,
                    generated_electrodograms,
                    real_electrodograms,
                    lambda_adv,
                    lambda_rec
                )

            gen_loss.backward()
            generator_optimizer.step()
            
            # Only train discriminator every N generator iterations
            if gen_iterations % config['gen_iterations'] == 0:
                discriminator_optimizer.zero_grad()

                with torch.no_grad():
                    generated_electrodograms = generator(noisy_audio)

                real_outputs = discriminator(real_electrodograms)
                fake_outputs = discriminator(generated_electrodograms)

                disc_loss, real_loss, fake_loss = discriminator_loss(
                    real_outputs,
                    fake_outputs,
                    lambda_adv
                )

                gp = compute_gradient_penalty(discriminator, real_electrodograms, generated_electrodograms, device=device)

                disc_loss += 10.0 * gp 
                
                disc_loss.backward()
                discriminator_optimizer.step()

                train_disc_losses.append(disc_loss.item())
                train_real_losses.append(real_loss.item())
                train_fake_losses.append(fake_loss.item())
            
            train_gen_losses.append(gen_loss.item())
            train_rec_losses.append(rec_loss.item())
            train_adv_losses.append(adv_loss.item())
            if config['LossWeights']['include_feature_matching']:
                train_match_losses.append(feature_matching_loss.item())
            
            train_loader_tqdm.set_postfix({
                'Rec. Loss': f'{rec_loss.item():.4f}',
                'Real. Loss': f'{real_loss.item():.4f}',
                'Fake. Loss': f'{fake_loss.item():.4f}'
            })

            gen_iterations += 1
            train_iter += 1

        train_gen_loss = np.mean(train_gen_losses)
        train_disc_loss = np.mean(train_disc_losses)
        train_real_loss = np.mean(train_real_losses)
        train_fake_loss = np.mean(train_fake_losses)
        train_rec_loss = np.mean(train_rec_losses)
        train_adv_loss = np.mean(train_adv_losses)
        train_match_loss = 0.0
        if config['LossWeights']['include_feature_matching']:
            train_match_loss = np.mean(train_match_losses)


        train_losses['Rec. Loss'].append(train_rec_loss)
        train_losses['Disc. Loss'].append(train_disc_loss)

        print(f"\033[1mEpoch {epoch+1}\033[0m")
        print("-" * 40)
        print(
            "\033[4mTraining Results:\033[0m\n"
            f"Generator Loss:\n"
            f"  Total: {train_gen_loss:.5f}\n"
            f"    - Rec. Loss: {train_rec_loss:.5f}\n"
            f"    - Adv. Loss: {train_adv_loss:.5f}\n"
            f"    - Feat. Matching: {train_match_loss:.5f}\n"
            "Discriminator Loss:\n"
            f"  Total: {train_disc_loss:.5f}\n"
            f"    - Real: {train_real_loss:.5f}\n"
            f"    - Fake: {train_fake_loss:.5f}"
        )
        
        
    
        if warmup_complete:
            # ---------------------
            #  Validation Loop
            # ---------------------
            val_gen_losses = []
            val_disc_losses = []
            val_fake_losses = []
            val_real_losses = []
            val_rec_losses = []
            val_adv_losses = []
            if config['LossWeights']['include_feature_matching']:
                val_match_losses = []

            generator.eval()
            discriminator.eval()
            
            with torch.no_grad():
                val_loader_tqdm = tqdm(
                    valid_data_loader,
                    desc=f'Validation Epoch {epoch + 1}',
                    leave=False,
                    dynamic_ncols=True
                )
                val_iter = 0
                for noisy_audio, real_electrodograms in val_loader_tqdm:
                    #if val_iter > 100:
                    #    break
                    noisy_audio = noisy_audio.to(device)
                    real_electrodograms = real_electrodograms.to(device)
                    generated_electrodograms = generator(noisy_audio)

                    if config['LossWeights']['include_feature_matching']:
                        real_outputs, real_features = discriminator(
                            real_electrodograms,
                            return_features=True
                        )
                        fake_outputs, fake_features = discriminator(
                            generated_electrodograms,
                            return_features=True
                        )
                        feature_matching_loss = 0
                        for fake_feat, real_feat in zip(fake_features, real_features):
                            feature_matching_loss += torch.nn.MSELoss()(fake_feat, real_feat)
                        feature_matching_loss /= len(fake_features)

                        gen_loss, rec_loss, adv_loss = generator_loss(
                            fake_outputs,
                            generated_electrodograms,
                            real_electrodograms,
                            lambda_adv,
                            lambda_rec
                        )
                        gen_loss = lambda_feat * feature_matching_loss + gen_loss
                    else:
                        real_outputs = discriminator(real_electrodograms)
                        fake_outputs = discriminator(generated_electrodograms)
                        gen_loss, rec_loss, adv_loss = generator_loss(
                            fake_outputs,
                            generated_electrodograms,
                            real_electrodograms,
                            lambda_adv,
                            lambda_rec
                        )

                    disc_loss, real_loss, fake_loss = discriminator_loss(
                        real_outputs,
                        fake_outputs,
                        lambda_adv
                    )

                    val_iter += 1

                    val_gen_losses.append(gen_loss.item())
                    val_disc_losses.append(disc_loss.item())
                    val_real_losses.append(real_loss.item())
                    val_fake_losses.append(fake_loss.item())
                    val_rec_losses.append(rec_loss.item())
                    val_adv_losses.append(adv_loss.item())
                    if config['LossWeights']['include_feature_matching']:
                        val_match_losses.append(feature_matching_loss.item())

            val_gen_loss = np.mean(val_gen_losses)
            val_disc_loss = np.mean(val_disc_losses)
            val_real_loss = np.mean(val_real_losses)
            val_fake_loss = np.mean(val_fake_losses)
            val_rec_loss = np.mean(val_rec_losses)
            val_adv_loss = np.mean(val_adv_losses)
            
            val_losses['Rec. Loss'].append(val_rec_loss)
            val_losses['Disc. Loss'].append(val_disc_loss)


            # Print results
            if config['LossWeights']['include_feature_matching']:
                val_match_loss = np.mean(val_match_losses)
                print(
                    "\033[\n4mValidation Results:\033[0m\n"
                    f"Generator Loss:\n"
                    f"  Total: {val_gen_loss:.5f}\n"
                    f"    - Rec. Loss: {val_rec_loss:.5f}\n"
                    f"    - Adv. Loss: {val_adv_loss:.5f}\n"
                    f"    - Feat. Matching: {val_match_loss:.5f}\n"
                    "Discriminator Loss:\n"
                    f"  Total: {val_disc_loss:.5f}\n"
                    f"    - Real: {val_real_loss:.5f}\n"
                    f"    - Fake: {val_fake_loss:.5f}"
                )
            else:
                print(
                    "\033[\n4mValidation Results:\033[0m\n"
                    f"Generator Loss:\n"
                    f"  Total: {val_gen_loss:.5f}\n"
                    f"  Rec. Loss: {val_rec_loss:.5f}\n"
                    f"  Adv. Loss: {val_adv_loss:.5f}\n"
                    "Discriminator Loss:\n"
                    f"  Total: {val_disc_loss:.5f}\n"
                    f"    - Real: {val_real_loss:.5f}\n"
                    f"    - Fake: {val_fake_loss:.5f}"
                )

            # ---------------------
            #  Early Stopping Check
            # ---------------------
            # Only count epochs after warmup
            if val_rec_loss < best_val_rec_loss:
                best_val_rec_loss = val_rec_loss
                epochs_no_improve = 0
                
                # Save model state
                torch.save({
                    'epoch': epoch + 1,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizerG_state_dict': generator_optimizer.state_dict(),
                    'optimizerD_state_dict': discriminator_optimizer.state_dict(),
                    'best_val_rec_loss': best_val_rec_loss,
                    'config': config 
                }, model_save_path)

                print("-" * 40)
                print("Model saved.")
            else:
                epochs_no_improve += 1
                print(f"Epochs without improvement: {epochs_no_improve}/{early_stop_patience}")

            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered. No improvement in validation loss for {early_stop_patience} epochs.")
                break
        
            generator_scheduler.step(val_rec_loss)
            discriminator_scheduler.step(val_disc_loss)

        else:
            # During warmup, do not update early stopping
            print("Skipping early stopping check and validation during warmup stabilization...\n")
            continue


        # Print current LR
        gen_lr = f"{generator_optimizer.param_groups[0]['lr']:.2e}"
        disc_lr = f"{discriminator_optimizer.param_groups[0]['lr']:.2e}"
        print(f"Generator learning rate: {gen_lr}")
        print(f"Discriminator learning rate: {disc_lr}")
        print_divider()

    # ---------------------
    #  Save Training Plots
    # ---------------------
    plot_loss(train_losses, val_losses, os.path.join(model_save_dir, 'training_progress.png'))
    train_losses_df = pd.DataFrame(train_losses)
    val_losses_df = pd.DataFrame(val_losses)
    train_losses_df.to_csv(os.path.join(model_save_dir, 'train_losses.csv'), index=False)
    val_losses_df.to_csv(os.path.join(model_save_dir, 'val_losses.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DeepGACE")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config file")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(config, args.config, timestamp)
