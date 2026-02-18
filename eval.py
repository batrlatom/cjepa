
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Any, Optional

# Local imports
from cjepa.utils.dist import is_main_process

def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    logger: Optional = None,
    mask_generator = None
) -> float:
    """
    Evaluate model on validation set.
    """
    model.eval()
    total_loss = 0.0
    total_history_loss = 0.0
    total_future_loss = 0.0
    num_steps = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            slots = batch["slots"].to(device, non_blocking=True)
            aux = batch["aux"].to(device, non_blocking=True)
            batch_size = slots.shape[0]
            
            # Predict
            # For validation, do we mask?
            # Typically validation should measure reconstruction/prediction error.
            # If we don't mask history, L_history is 0 (or undefined).
            # We should probably mask similar to training to measure performance.
            # Or mask *only* future for pure prediction evaluation.
            
            mask_indices = None
            if mask_generator:
                mask_indices = mask_generator(batch_size=batch_size, device=device)
            
            outputs = model(slots, aux=aux, mask_indices=mask_indices)
            pred_slots = outputs["pred_slots"]
            
            losses = criterion(pred_slots, slots, mask_indices)
            
            loss = losses["loss_total"]
            total_loss += loss.item()
            total_history_loss += losses["loss_history"].item()
            total_future_loss += losses["loss_future"].item()
            num_steps += 1
            
    avg_loss = total_loss / num_steps
    avg_hist_loss = total_history_loss / num_steps
    avg_future_loss = total_future_loss / num_steps
    
    if is_main_process() and logger:
        logger.log_scalar("Val/Loss_Total", avg_loss, epoch)
        logger.log_scalar("Val/Loss_History", avg_hist_loss, epoch)
        logger.log_scalar("Val/Loss_Future", avg_future_loss, epoch)
        
    return avg_loss
