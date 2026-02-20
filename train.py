import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from loss import CJEPALoss
from model import CJEPA, CJEPAConfig
from dataset_sudoku import SudokuDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_object_mask(B: int, N: int, m_min: int, m_max: int, device: torch.device) -> torch.Tensor:
    M = torch.zeros((B, N), dtype=torch.bool, device=device)
    for b in range(B):
        k = random.randint(m_min, m_max)
        if k > 0:
            indices = torch.randperm(N, device=device)[:k]
            M[b, indices] = True
    return M


def evaluate_mask_acc(model: CJEPA, loader: DataLoader, cfg: CJEPAConfig, device: torch.device) -> float:
    model.eval()
    total_acc = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            z = batch["z"].to(device)
            B = z.shape[0]

            # Mask 40 cells for evaluation
            M = sample_object_mask(B, cfg.N, 40, 40, device)
            out = model(z=z, M=M)
            z_hat = out["z_hat"]

            preds = z_hat.argmax(dim=-1)
            correct = (preds == z)
            mask_acc = (correct.float() * M.unsqueeze(1).float()).sum() / M.sum().clamp_min(1).float()
            
            total_acc += float(mask_acc.cpu())
            n_batches += 1

    model.train()
    return total_acc / max(1, n_batches)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_schedule: str,
    warmup_epochs: int,
    epochs: int,
    steps_per_epoch: int,
):
    if lr_schedule == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def split_indices_random(n: int, val_ratio: float, seed: int) -> Tuple[list, list]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(n * val_ratio)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Train C-JEPA on Sudoku")
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--exp_name", type=str, default="sudoku")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--mask_min", type=int, default=10)
    parser.add_argument("--mask_max", type=int, default=60)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dataset_size", type=int, default=100000)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--tb_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(args.output_dir) / args.exp_name
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    run_numbers = []
    for d in existing_runs:
        try:
            run_numbers.append(int(d.name.split("_")[1]))
        except ValueError:
            pass
    next_run = max(run_numbers) + 1 if run_numbers else 0
    output_dir = base_dir / f"run_{next_run}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(args.tb_dir) if args.tb_dir is not None else (output_dir / "tb")
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    dataset = SudokuDataset(size=args.dataset_size)
    train_idx, val_idx = split_indices_random(len(dataset), val_ratio=args.val_ratio, seed=args.seed)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    cfg = CJEPAConfig(
        N=81,
        d=10,
        D=args.model_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        T_h=1,
        T_p=0,
    )

    model = CJEPA(cfg).to(device)
    criterion = CJEPALoss(T_h=cfg.T_h).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer=opt,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        epochs=args.epochs,
        steps_per_epoch=max(1, len(train_loader)),
    )

    run_cfg = {
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "seed": args.seed,
        "tb_dir": str(tb_dir),
        "model": asdict(cfg),
        "train_args": vars(args),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    print(f"device={device} train={len(train_idx)} val={len(val_idx)}")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in train_loader:
            z = batch["z"].to(device)
            B = z.shape[0]

            M = sample_object_mask(B=B, N=cfg.N, m_min=args.mask_min, m_max=args.mask_max, device=device)
            out = model(z=z, M=M)
            losses = criterion(z_hat=out["z_hat"], z_target=z, mask_map=out["mask_map"])

            opt.zero_grad(set_to_none=True)
            losses["L_mask"].backward()
            opt.step()
            scheduler.step()

            total_loss += float(losses["L_mask"].detach().cpu())
            total_acc += float(losses["mask_acc"].detach().cpu())
            n_batches += 1

            writer.add_scalar("train/L_mask", float(losses["L_mask"].detach().cpu()), global_step)
            writer.add_scalar("train/mask_acc", float(losses["mask_acc"].detach().cpu()), global_step)
            writer.add_scalar("train/lr", float(opt.param_groups[0]["lr"]), global_step)
            global_step += 1

        train_loss = total_loss / max(1, n_batches)
        train_acc = total_acc / max(1, n_batches)
        val_mask_acc = evaluate_mask_acc(model, val_loader, cfg, device)
        writer.add_scalar("epoch/train_L_mask", train_loss, epoch)
        writer.add_scalar("epoch/train_mask_acc", train_acc, epoch)
        writer.add_scalar("epoch/val_mask_acc", val_mask_acc, epoch)

        print(f"epoch={epoch:03d} train_L_mask={train_loss:.6f} train_mask_acc={train_acc:.4f} val_mask_acc={val_mask_acc:.4f}")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": asdict(cfg),
                    "epoch": epoch,
                    "train_L_mask": train_loss,
                    "val_mask_acc": val_mask_acc,
                },
                ckpt_path,
            )

    writer.close()


if __name__ == "__main__":
    main()
