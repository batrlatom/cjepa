
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class CJEPALoss(nn.Module):
    """
    Computes reconstruction loss for masked slots.
    L = L_history + L_future
    """
    def __init__(self, history_len: int):
        super().__init__()
        self.history_len = history_len

    def forward(self, 
                pred_slots: torch.Tensor, 
                target_slots: torch.Tensor, 
                mask_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_slots: (B, T, N, D)
            target_slots: (B, T, N, D)
            mask_indices: (B, N) boolean. True if object is masked in history.
        """
        B, T, N, D = pred_slots.shape
        
        # Calculate MSE per element
        # (B, T, N)
        mse = F.mse_loss(pred_slots, target_slots, reduction='none').mean(dim=-1)
        
        # History Loss
        # Mask condition: t < history_len AND object in mask_indices AND t > 0
        # (We don't compute loss on t=0 anchor because it's given)
        
        # Create masks
        time_indices = torch.arange(T, device=pred_slots.device).reshape(1, T, 1) # (1, T, 1)
        
        if mask_indices.dim() == 3: # (B, T, N) provided by dataset
             history_mask = mask_indices
             # Dataset mask usually implies "This is masked", whether history or future.
             # But here we separate History/Future.
             # In Action Maze, everything is "History" (T < history_len).
             # So we use mask_indices as history_mask directly?
             # Yes, config.history_len = total_len.
        else:
            # Standard random masking logic
            # History Mask
            # t in [1, H-1]
            is_history_time = (time_indices > 0) & (time_indices < self.history_len)
            # Object is masked
            is_obj_masked = mask_indices.unsqueeze(1) # (B, 1, N)
            
            history_mask = is_history_time & is_obj_masked # (B, T, N)
        
        # Future Loss
        # t >= H
        # All objects are masked in future
        is_future_time = time_indices >= self.history_len
        future_mask = is_future_time.expand(B, T, N) # (B, T, N)
        
        # Compute losses
        loss_history = (mse * history_mask.float()).sum() / (history_mask.sum().float() + 1e-6)
        loss_future = (mse * future_mask.float()).sum() / (future_mask.sum().float() + 1e-6)
        
        loss_total = loss_history + loss_future
        
        return {
            "loss_total": loss_total,
            "loss_history": loss_history,
            "loss_future": loss_future
        }
