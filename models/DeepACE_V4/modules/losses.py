import torch.nn as nn

class LossFunctionSelector:
    def __init__(self):
        self.loss_functions = {
            "L1": self._get_l1_loss,
            "MSE": self._get_mse_loss
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

