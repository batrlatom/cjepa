
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math

def get_optimizer(model, config):
    """
    Build AdamW optimizer.
    """
    # Separate weight decay for bias/layernorm parameter if desired
    # For now, simple standard AdamW
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    return optimizer

def get_scheduler(optimizer, config, steps_per_epoch):
    """
    Cosine Learning Rate Scheduler with Warmup.
    """
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler
