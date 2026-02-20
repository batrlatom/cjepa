import torch
import torch.nn as nn


class CJEPALoss(nn.Module):
    """
    Minimal C-JEPA objective.

    Eq. (5):
      L_mask = E[ 1[z_bar_tau^i != z_tau^i] * ||z_hat_tau^i - z_tau^i||_2^2 ]

    Eq. (6) decomposition:
      L_mask = L_history + L_future
    """

    def __init__(self, T_h: int):
        super().__init__()
        self.T_h = T_h

    def forward(self, z_hat: torch.Tensor, z_target: torch.Tensor, mask_map: torch.Tensor) -> dict:
        if z_hat.shape != z_target.shape:
            raise ValueError(f"Shape mismatch: z_hat={tuple(z_hat.shape)} vs z_target={tuple(z_target.shape)}")

        B, T, N, _ = z_hat.shape
        z_target = z_target.detach()

        sq_err = (z_hat - z_target).pow(2).mean(dim=-1)  # (B, T, N)
        tau = torch.arange(T, device=z_hat.device).view(1, T, 1)

        history_selector = mask_map & (tau < self.T_h)
        future_selector = tau >= self.T_h

        L_history = (sq_err * history_selector.float()).sum() / history_selector.sum().clamp_min(1).float()
        L_future = (sq_err * future_selector.float()).sum() / future_selector.sum().clamp_min(1).float()

        # Eq. (5) exact masked-token objective.
        L_mask_eq5 = (sq_err * mask_map.float()).sum() / mask_map.sum().clamp_min(1).float()

        L_mask = L_history + L_future
        return {
            "L_mask": L_mask,
            "L_history": L_history,
            "L_future": L_future,
            "L_mask_eq5": L_mask_eq5,
        }
