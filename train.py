
import torch
from torch.cuda.amp import autocast, GradScaler
import logging
from tqdm import tqdm
import time
from typing import Dict, Optional

# Local imports
from cjepa.utils.dist import is_main_process

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: GradScaler,
    epoch: int,
    logger: Optional = None,
    mask_generator = None,
    log_freq: int = 10
):
    """
    Train for one epoch.
    """
    model.train()
    
    total_loss = 0.0
    num_steps = len(dataloader)
    
    start_time = time.time()
    
    pbar = tqdm(enumerate(dataloader), total=num_steps, desc=f"Epoch {epoch}", disable=not is_main_process())
    for step, batch in pbar:
        # Move data to device
        slots = batch["slots"].to(device, non_blocking=True) # (B, T, N, D)
        aux = batch["aux"].to(device, non_blocking=True)     # (B, T, D_aux)
        
        batch_size = slots.shape[0]
        
        # Check if dataset provides mask
        if "mask_indices" in batch:
            mask_indices = batch["mask_indices"].to(device, non_blocking=True)
            # Ensure bool
            if mask_indices.dtype != torch.bool:
                mask_indices = mask_indices.bool()
        else:
            # Generate random mask
            mask_indices = mask_generator(batch_size=batch_size, device=device)
        
        optimizer.zero_grad()
        
        # Mixed Precision Forward
        with autocast():
            # Model Forward
            outputs = model(slots, aux=aux, mask_indices=mask_indices)
            pred_slots = outputs["pred_slots"]
            
            # Loss Calculation
            loss_dict = criterion(pred_slots, slots, mask_indices)
            loss = loss_dict["loss_total"]
            
        # Backward & Step
        scaler.scale(loss).backward()
        
        # Clip Gradient (Unscale first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Configurable
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        total_loss += loss.item()
        
        if is_main_process() and logger and step % log_freq == 0:
            global_step = epoch * num_steps + step
            logger.log_scalar("Train/Loss_Total", loss.item(), global_step)
            logger.log_scalar("Train/Loss_History", loss_dict["loss_history"].item(), global_step)
            logger.log_scalar("Train/Loss_Future", loss_dict["loss_future"].item(), global_step)
            logger.log_scalar("Train/LR", current_lr, global_step)
            
            # Throughput
            elapsed = time.time() - start_time
            throughput = (step + 1) * batch_size / elapsed
            logger.log_scalar("System/Throughput", throughput, global_step)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_steps
    return avg_loss
