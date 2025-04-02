import torch

def si_sdr_loss(estimate, target, eps=1e-8):
    estimate = estimate.squeeze(1)
    target = target.squeeze(1)

    target_mean = torch.mean(target, dim=-1, keepdim=True)
    estimate_mean = torch.mean(estimate, dim=-1, keepdim=True)
    target_zero_mean = target - target_mean
    estimate_zero_mean = estimate - estimate_mean

    scaling_factor = torch.sum(estimate_zero_mean * target_zero_mean, dim=-1, keepdim=True) / \
                     (torch.sum(target_zero_mean ** 2, dim=-1, keepdim=True) + eps)
    
    projection = scaling_factor * target_zero_mean

    noise = estimate_zero_mean - projection
    sdr = 10 * torch.log10((torch.sum(projection ** 2, dim=-1) + eps) /
                           (torch.sum(noise ** 2, dim=-1) + eps))

    return -torch.mean(sdr)

def snr_loss(estimate, target, eps=1e-8):
    """
    Compute the negative Signal-to-Noise Ratio (SNR) loss between estimate and target.
    
    Args:
        estimate (Tensor): Estimated signal, shape [batch, 1, time] or [batch, time]
        target (Tensor): Target signal, shape [batch, 1, time] or [batch, time]
        eps (float): Small constant for numerical stability
        
    Returns:
        Tensor: Negative mean SNR across the batch (higher is better for optimization)
    """
    # Ensure the tensors are 2D (batch, time)
    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
        
    # Calculate signal power (target)
    signal_power = torch.sum(target ** 2, dim=-1) + eps
    
    # Calculate noise (difference between estimate and target)
    noise = estimate - target
    noise_power = torch.sum(noise ** 2, dim=-1) + eps
    
    # Compute SNR in dB
    snr = 10 * torch.log10(signal_power / noise_power)
    
    # Return negative mean SNR (for minimization in training)
    return -torch.mean(snr)