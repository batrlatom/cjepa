import torch
import torch.nn as nn
import torch.nn.functional as F

class CJEPALoss(nn.Module):
    """
    Sudoku-adapted C-JEPA objective using CrossEntropyLoss over masked positions.
    """

    def __init__(self, T_h: int = 1):
        super().__init__()
        self.T_h = T_h

    def forward(self, z_hat: torch.Tensor, z_target: torch.Tensor, mask_map: torch.Tensor) -> dict:
        # z_hat: (B, T, N, C) logits
        # z_target: (B, T, N) integer class indices
        B, T, N, C = z_hat.shape
        z_target = z_target.detach()

        z_hat_flat = z_hat.view(-1, C)
        z_target_flat = z_target.view(-1)

        loss_all = F.cross_entropy(z_hat_flat, z_target_flat, reduction='none').view(B, T, N)

        # Average over masked tokens
        L_mask = (loss_all * mask_map.float()).sum() / mask_map.sum().clamp_min(1).float()

        # Calculate accuracy on masked tokens
        preds = z_hat.argmax(dim=-1)
        correct = (preds == z_target)
        mask_acc = (correct.float() * mask_map.float()).sum() / mask_map.sum().clamp_min(1).float()

        return {
            "L_mask": L_mask,
            "mask_acc": mask_acc,
        }
