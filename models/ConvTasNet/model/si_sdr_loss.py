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