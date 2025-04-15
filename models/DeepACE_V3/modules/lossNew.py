import torch
import torch.nn as nn

def snr_improvement(generated, noisy, real, eps=1e-8):
    """
    Computes SNR improvement between generated and real electrodograms.
    """
    generated_lin = torch.exp(generated)
    noisy_lin = torch.exp(noisy)
    real_lin = torch.exp(real)

    noise_gen = real_lin - generated_lin
    noise_noisy = real_lin - noisy_lin

    gen_power = (generated_lin ** 2).mean(dim=(-2, -1)).clamp(min=eps)
    noise_power_gen = (noise_gen ** 2).mean(dim=(-2, -1)).clamp(min=eps)
    noise_power_noisy = (noise_noisy ** 2).mean(dim=(-2, -1)).clamp(min=eps)

    snr_gen = gen_power / noise_power_gen
    snr_noisy = gen_power / noise_power_noisy

    snr_gain = (snr_gen / snr_noisy).clamp(min=eps).log10()

    return -snr_gain.mean()  # Minimize negative gain to maximize SNR improvement

class LinearCrossCorrelation(nn.Module):
    def __init__(self):
        super(LinearCrossCorrelation, self).__init__()

    def forward(self, input, target):
        input_mean = input.mean(dim=(-2, -1), keepdim=True)
        target_mean = target.mean(dim=(-2, -1), keepdim=True)

        input_centered = input - input_mean
        target_centered = target - target_mean

        covariance = (input_centered * target_centered).mean(dim=(-2, -1))
        input_std = input_centered.var(dim=(-2, -1), unbiased=False).sqrt().clamp(min=1e-8)
        target_std = target_centered.var(dim=(-2, -1), unbiased=False).sqrt().clamp(min=1e-8)

        correlation = covariance / (input_std * target_std)
        correlation = torch.clamp(correlation, -1.0, 1.0)

        return 1 - correlation.abs().mean()  

def generator_loss(fake_output, generated, real, noisy, lambda_adv=1.0, lambda_rec=1.0, lambda_per=1.0):
    """
    Refined generator loss with adversarial, perceptual, and SNR components.
    """
    adv_loss = -torch.mean(fake_output)  # Hinge loss instead of MSE

    mse_loss = nn.MSELoss()(generated, real)
    cc_loss = LinearCrossCorrelation()(generated, real)
    snr_loss = snr_improvement(generated, noisy, real)

    total_loss = lambda_adv * adv_loss + lambda_rec * mse_loss + lambda_per * (cc_loss + snr_loss)
    
    return total_loss, mse_loss, adv_loss, snr_loss


def discriminator_loss(real_output, fake_output, lambda_adv=1.0):
    """
    Discriminator hinge loss.
    """
    real_loss = torch.mean(nn.ReLU()(1.0 - real_output))
    fake_loss = torch.mean(nn.ReLU()(1.0 + fake_output))

    total_loss = lambda_adv * 0.5 * (real_loss + fake_loss)

    return total_loss, real_loss, fake_loss