
import os
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import logging

class Logger:
    def __init__(self, log_dir, enabled=True):
        self.enabled = enabled
        self.writer = None
        if enabled:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            
    def log_scalar(self, tag, value, step):
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)
            
    def log_metrics(self, metrics, step):
        if self.enabled and self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)
                
    def close(self):
        if self.writer:
            self.writer.close()

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
        
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch
    }
    torch.save(state, path)
    logging.info(f"Checkpoint saved to {path}")

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    if not os.path.exists(path):
        logging.warning(f"Checkpoint not found at {path}")
        return 0
        
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    if scheduler and 'scheduler' in checkpoint and checkpoint['scheduler']:
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    logging.info(f"Loaded checkpoint from {path} (epoch {epoch})")
    return epoch
