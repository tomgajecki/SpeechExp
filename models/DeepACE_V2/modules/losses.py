import torch
import torch.nn as nn

class LinearCrossCorrelation(nn.Module):
    def __init__(self):
        super(LinearCrossCorrelation, self).__init__()

    def forward(self, input, target):
        assert input.shape == target.shape, "Input and target must have the same shape"

        # Compute means along the last dimension
        input_mean = input.mean(dim=-1, keepdim=True)
        target_mean = target.mean(dim=-1, keepdim=True)

        # Center the inputs
        input_centered = input - input_mean
        target_centered = target - target_mean

        # Compute covariance and standard deviations
        covariance = (input_centered * target_centered).mean(dim=-1)
        input_std = input_centered.pow(2).mean(dim=-1).sqrt().clamp(min=1e-8)
        target_std = target_centered.pow(2).mean(dim=-1).sqrt().clamp(min=1e-8)

        # Compute correlation coefficient
        correlation = covariance / (input_std * target_std)
        correlation = torch.clamp(correlation, -1.0, 1.0)

        # Return loss
        return (1 - correlation.abs()).mean()


class LossFunctionSelector:
    def __init__(self):
        self.loss_functions = {
            "L1": self._get_l1_loss,
            "MSE": self._get_mse_loss,
            "weighted": self._get_weighted_loss
        }

    def get_loss(self, loss_name, **kwargs):
        if loss_name in self.loss_functions:
            return self.loss_functions[loss_name](**kwargs)
        else:
            raise ValueError(f"Loss '{loss_name}' not found.")

    def _get_l1_loss(self):
        return nn.L1Loss()

    def _get_mse_loss(self):
        return nn.MSELoss()

    def _get_weighted_loss(self):
        return AutoWeightedFeatureLoss()

class WeightedFeatureLoss(nn.Module):
    def __init__(self, weights=torch.tensor([0.5] * 11 + [1.0] * 11)):
        """
        Initialize the WeightedFeatureLoss with the given weights.

        Args:
            weights (torch.Tensor): A 1D tensor of shape (num_features,) containing the weights for each feature.
        """
        super(WeightedFeatureLoss, self).__init__()
        self.weights = weights

    def forward(self, input, target):
        """
        Compute the weighted loss between input and target.

        Args:
            input (torch.Tensor): The predicted output of shape (batch, features, time_steps).
            target (torch.Tensor): The ground truth of shape (batch, features, time_steps).

        Returns:
            torch.Tensor: The computed weighted loss.
        """
        # Ensure weights match the feature dimension
        assert input.shape[1] == self.weights.shape[0], "Weights should have the same number of elements as the feature dimension."

        # Compute squared differences
        loss_per_feature = (input - target) ** 2

        # Apply weights to each feature dimension
        weighted_loss = loss_per_feature * self.weights.view(1, -1, 1).to(input.device)
        # Compute the mean loss over all dimensions
        return weighted_loss.mean()



class AutoWeightedFeatureLoss(nn.Module):
    def __init__(self, eps=1e-8, momentum=0.9):
        super(AutoWeightedFeatureLoss, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_error', torch.zeros(22))

    def forward(self, input, target):
        loss_per_feature = (input - target) ** 2  # Shape: (batch, features, time_steps)
        error_per_feature = loss_per_feature.mean(dim=(0, 2))  # Shape: (features,)

        # Ensure device compatibility
        error_per_feature = error_per_feature.to(self.running_error.device)

        # Update running_error without tracking gradients
        with torch.no_grad():
            self.running_error.mul_(self.momentum)
            self.running_error.add_((1 - self.momentum) * error_per_feature.detach())

        # Compute weights
        weights = self.running_error / (self.running_error.sum() + self.eps)
        weights = weights.detach()
        weights = weights.view(1, -1, 1).to(input.device)
        weighted_loss = loss_per_feature * weights
        return weighted_loss.mean()