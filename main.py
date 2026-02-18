
import os
import argparse
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from cjepa.config import Config, ModelConfig, TrainConfig, DataConfig, NLPConfig
from cjepa.model import CJEPA
from cjepa.loss import CJEPALoss
from cjepa.dataset import PushTDataset
from cjepa.dataset_nlp import NLPDataset
from cjepa.utils.dist import setup_dist, cleanup_dist, setup_seed, is_main_process
from cjepa.utils.logging import Logger, save_checkpoint, load_checkpoint
from cjepa.optim import get_optimizer, get_scheduler
from cjepa.augment import generate_object_mask
from cjepa.train import train_one_epoch
from cjepa.eval import evaluate

def main():
    parser = argparse.ArgumentParser(description="C-JEPA Training")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick sanity check")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--save_freq", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--dataset", type=str, default="pusht", choices=["pusht", "nlp", "reasoning", "maze", "dynamic_maze"], help="Dataset type")
    parser.add_argument("--dummy_data", action="store_true", help="Use dummy data")
    
    args = parser.parse_args()
    
    # Setup Distributed
    dist_params = setup_dist()
    device = torch.device(f"cuda:{dist_params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    setup_seed(42 + dist_params["rank"])
    
    # Config
    cfg = Config()
    cfg.train.batch_size = args.batch_size
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.save_freq = args.save_freq
    cfg.train.output_dir = args.output_dir
    cfg.dataset_type = args.dataset
    
    if args.dataset == "nlp":
        # Override model config with NLP defaults
        nlp_defaults = NLPConfig()
        cfg.model = nlp_defaults
        print("Using NLP Configuration.")
    elif args.dataset == "reasoning":
        from cjepa.config import ReasoningConfig
        cfg.model = ReasoningConfig()
        cfg.model.slot_dim = 64 # Override if needed, but class default wins
        
        # Override Train defaults for this task
        cfg.train.epochs = 200 # Need more for depth 16 generalization
        cfg.train.batch_size = 64
        cfg.train.save_freq = 10
        cfg.train.mask_ratio_max = 0 # Disable history masking, only future prediction
        print("Using Reasoning Configuration.")
    elif args.dataset == "maze":
        from cjepa.config import MazeConfig
        cfg.model = MazeConfig()
        # Override Train defaults
        cfg.train.epochs = 100
        cfg.train.batch_size = 64
        cfg.train.save_freq = 10
        cfg.train.mask_ratio_max = 0 # No object masking, just future prediction
        print("Using Maze Configuration.")
    elif args.dataset == "dynamic_maze":
        from cjepa.config import DynamicMazeConfig
        cfg.model = DynamicMazeConfig()
        cfg.train.epochs = 50
        cfg.train.batch_size = 64
        cfg.train.save_freq = 5
        # mask_ratio_max doesn't matter because dataset provides masks
        print("Using Dynamic Maze Configuration.")
    if args.dummy_data or args.dry_run:
        cfg.data.dummy_mode = True
        
    if args.dry_run:
        cfg.train.epochs = 1
        cfg.data.dummy_batch_size = 4
        cfg.train.batch_size = 4
        cfg.train.log_freq = 1
        cfg.train.num_workers = 0
        print("Running in DRY RUN mode.")

    # Data
    if cfg.dataset_type == "nlp":
        train_dataset = NLPDataset(cfg.model, split="train", download=True)
        # Force single process for demo stability
        cfg.train.num_workers = 0
    elif cfg.dataset_type == "reasoning":
        from cjepa.dataset_reasoning import ReasoningDataset
        train_dataset = ReasoningDataset(cfg.model, split="train")
        cfg.train.num_workers = 0 # Synthetic generator is fast enough
    elif cfg.dataset_type == "maze":
        from cjepa.dataset_maze import MazeDataset
        train_dataset = MazeDataset(cfg.model, split="train")
        cfg.train.num_workers = 0
    elif cfg.dataset_type == "dynamic_maze":
        from cjepa.dataset_dynamic_maze import DynamicMazeDataset
        train_dataset = DynamicMazeDataset(cfg.model, split="train")
        cfg.train.num_workers = 0
    else:
        train_dataset = PushTDataset(cfg.data, split="train")
    if is_main_process():
        print(f"Train Dataset Size: {len(train_dataset)}")
    
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True) if dist.is_initialized() else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory
    )
    
    # Model
    model = CJEPA(cfg.model).to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[dist_params["gpu"]])
        
    # Optimizer & Scheduler
    optimizer = get_optimizer(model, cfg.train)
    steps_per_epoch = len(train_loader)
    scheduler = get_scheduler(optimizer, cfg.train, steps_per_epoch)
    
    # Loss
    criterion = CJEPALoss(history_len=cfg.model.history_len).to(device)
    
    # Scaler for AMP
    scaler = torch.cuda.amp.GradScaler()
    
    # Logger
    logger = None
    if is_main_process():
        logger = Logger(cfg.train.output_dir)
        
    # Validation Dataset (Optional/Placeholder)
    # Using train set subset for dry run simplicity
    
    # Training Loop
    if is_main_process():
        print("Starting training...")
        
    for epoch in range(cfg.train.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
            
        # Mask Generator wrapper
        mask_gen = lambda batch_size, device: generate_object_mask(
            batch_size, cfg.model.num_slots, 
            cfg.train.mask_ratio_min, cfg.train.mask_ratio_max, 
            device
        )
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, 
            device, scaler, epoch, logger, mask_gen, cfg.train.log_freq
        )
        
        if is_main_process():
            print(f"Epoch {epoch}: Train Loss {train_loss:.4f}")
            
            if (epoch + 1) % cfg.train.save_freq == 0:
                save_path = os.path.join(cfg.train.output_dir, f"checkpoint_epoch_{epoch}.pth")
                save_checkpoint(model, optimizer, scheduler, epoch, save_path)

    if is_main_process():
        if logger:
            logger.close()
        print("Training complete.")
        
    cleanup_dist()

if __name__ == "__main__":
    main()
