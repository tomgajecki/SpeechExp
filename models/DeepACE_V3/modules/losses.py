import torch
import torch.nn as nn

def generator_loss(fake_output, generated, real, lambda_adv=0.01, lambda_rec=1.0):
    # 1. Compute MSE map (electrode-wise errors)
    mse_map = (generated - real).pow(2)  # [batch, electrodes, time]
    
    # 2. Create adaptive mask for well-reconstructed regions
    with torch.no_grad():
        # Calculate per-sample MSE
        sample_mse = mse_map.mean(dim=(1,2))  # [batch]
        
        # Create batch-wise mask (sigmoid transition at threshold)
        mask = torch.sigmoid(10 * (0.05 - sample_mse))  # [batch]
    
    # 3. Reconstruction loss (full weight)
    recon_loss = torch.mean(mse_map)
    
    # 4. Adversarial loss handling multi-output discriminator
    adv_loss = 0

    adv_loss += torch.mean(mask * torch.nn.functional.softplus(-fake_output.squeeze()))
    
    # 5. Combine losses with dynamic weighting
    total_loss = (
        lambda_rec * recon_loss +
        lambda_adv * adv_loss * torch.sigmoid(10 * (0.05 - recon_loss.detach()))
    )
    
    return total_loss, recon_loss, adv_loss

def discriminator_loss(real_output, fake_output, lambda_adv=1.0):
    """
    Discriminator hinge loss.
    """
    real_loss = torch.mean(nn.ReLU()(1.0 - real_output))
    fake_loss = torch.mean(nn.ReLU()(1.0 + fake_output))

    total_loss = lambda_adv * 0.5 * (real_loss + fake_loss)

    return total_loss, real_loss, fake_loss

