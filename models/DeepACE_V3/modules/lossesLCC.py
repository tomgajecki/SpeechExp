import torch
import torch.nn as nn


def snr_improvement(generated, noisy, eps=1e-8):
    """
    Computes SNR improvement between two log-compressed electrodograms by:
      1) Exponentiating them back to linear scale
      2) Computing standard SNR improvement in the linear domain
      3) Returning the difference (in dB)

    Args:
        generated_log (Tensor): Generated electrodograms in log domain
        noisy_log (Tensor): Noisy electrodograms in log domain
        eps (float): Small constant to prevent division by zero
    
    Returns:
        snr_improvement (Tensor): Scalar value of SNR improvement in dB
    """
    # 1) Convert from log domain to linear domain
    generated_lin = torch.exp(generated)  # linear scale
    noisy_lin = torch.exp(noisy)          # linear scale

    # 2) Compute the "noise" in linear domain
    noise_lin = noisy_lin - generated_lin

    # 3) Compute power of generated and noise signals (linear scale)
    gen_power = (generated_lin ** 2).mean(dim=(-2, -1)).clamp(min=eps)
    noise_power = (noise_lin ** 2).mean(dim=(-2, -1)).clamp(min=eps)

    # 4) Compute SNRs in the linear domain
    snr_gen = gen_power / noise_power

    # 5) Convert to dB and get improvement
    #    10 * log10(...) is typical for power ratios.
    snr = 10.0 * snr_gen.clamp(min=eps).log10()
    

    # 6) Return the mean improvement in dB
    return -snr.mean()


def generator_loss(
    fake_output, 
    generated_electrodograms, 
    real_electrodograms, 
    noisy_electrodograms, 
    lambda_adv=1.0, 
    lambda_rec=1.0,
    lambda_per=1.0,
):
    """
    Generator loss consists of:
      - Adversarial loss: Fake output should be classified as real.
      - Reconstruction loss (MSE) between generated and real electrodograms.
      - Linear cross-correlation loss to encourage high correlation.
      - SNR improvement as a regularization term.
    """
    # Adversarial Loss
    adv_loss = torch.mean((fake_output - 1) ** 2)
    
    # Reconstruction Loss (MSE)
    mse_loss = nn.MSELoss()(generated_electrodograms, real_electrodograms)
    
    # Linear Cross-Correlation Loss (fixed)tl
    cc_loss = LinearCrossCorrelation()(generated_electrodograms, real_electrodograms)

    # SNR Improvement Loss (fixed)
    snr_loss = snr_improvement(generated_electrodograms, noisy_electrodograms)


    # Multiply and negate the product so that it gets maximized
    #per_loss = cc_loss + snr_loss
    total_loss = lambda_adv * adv_loss + lambda_rec * mse_loss + lambda_per * snr_loss

    return total_loss, snr_loss, adv_loss, snr_loss


class LinearCrossCorrelation(nn.Module):
    def __init__(self):
        super(LinearCrossCorrelation, self).__init__()

    def forward(self, input, target):
        assert input.shape == target.shape, "Input and target must have the same shape"

        input_mean = input.mean(dim=(-2, -1), keepdim=True)
        target_mean = target.mean(dim=(-2, -1), keepdim=True)

        input_centered = input - input_mean
        target_centered = target - target_mean

        covariance = (input_centered * target_centered).mean(dim=(-2, -1))
        input_std = input_centered.var(dim=(-2, -1), unbiased=False).sqrt().clamp(min=1e-8)
        target_std = target_centered.var(dim=(-2, -1), unbiased=False).sqrt().clamp(min=1e-8)

        correlation = covariance / (input_std * target_std)
        correlation = torch.clamp(correlation, -1.0, 1.0)

        return 1 - correlation.abs().mean()  # Fix: Encourage high correlation by minimizing loss


def discriminator_loss(real_output, fake_output, lambda_adv=1.0, training=True):
    """
    Computes the discriminator BCE loss.
    """
    real_labels = torch.ones_like(real_output) * 0.9  # Label smoothing
    fake_labels = torch.zeros_like(fake_output) + 0.1  # Label smoothing

    real_loss = torch.mean((real_output - real_labels) ** 2)
    fake_loss = torch.mean((fake_output - fake_labels) ** 2)
    total_loss = lambda_adv * 0.5 * (real_loss + fake_loss)

    return total_loss, real_loss, fake_loss
