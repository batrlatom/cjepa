import torch
import torch.nn as nn


class ActionRegressionLoss(nn.Module):
    """Simple MSE loss for normalized action trajectory regression."""

    def __init__(self):
        super().__init__()
        self._mse = nn.MSELoss()

    def forward(self, pred_actions: torch.Tensor, target_actions: torch.Tensor) -> dict:
        loss = self._mse(pred_actions, target_actions)
        return {"mse": loss}
