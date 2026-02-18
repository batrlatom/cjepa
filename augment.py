
import torch
import numpy as np

def generate_object_mask(
    batch_size: int,
    num_slots: int,
    mask_ratio_min: int,
    mask_ratio_max: int,
    device: torch.device
) -> torch.Tensor:
    """
    Generates a mask for object slots.
    Args:
        batch_size: B
        num_slots: N
        mask_ratio_min: min objects to mask
        mask_ratio_max: max objects to mask (inclusive)
    
    Returns:
        mask: (B, N) boolean tensor where True indicates MASKED.
    """
    # Sample number of masked objects per sample
    # Using simple uniform sampling for now
    num_masked = torch.randint(mask_ratio_min, mask_ratio_max + 1, (batch_size,), device=device)
    
    mask = torch.zeros((batch_size, num_slots), dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        # Randomly select indices to mask
        indices = torch.randperm(num_slots, device=device)[:num_masked[i]]
        mask[i, indices] = True
        
    return mask

def get_mask_tokens(input_shape, mask):
    """
    Helper to create mask tokens matching shape.
    Args:
        input_shape: (B, T, N, D)
        mask: (B, N)
    """
    # This logic belongs in the model forward pass usually,
    # as we need learnable mask tokens.
    pass
