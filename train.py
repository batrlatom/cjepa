import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

from loss import CJEPALoss
from model import CJEPA, CJEPAConfig, sample_object_mask


class PushTTensorDataset(Dataset):
    def __init__(self, slots: torch.Tensor, aux: torch.Tensor):
        self.slots = slots
        self.aux = aux

    def __len__(self) -> int:
        return self.slots.shape[0]

    def __getitem__(self, idx: int):
        return {
            "z": self.slots[idx],
            "u": self.aux[idx],
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "cjepa" / "data" / "pusht_slots.pt"


def build_config_from_data(
    slots: torch.Tensor,
    aux: torch.Tensor,
    meta: dict,
    model_dim: int,
    n_layers: int,
    n_heads: int,
) -> CJEPAConfig:
    T_total = int(slots.shape[1])
    N = int(slots.shape[2])
    d = int(slots.shape[3])
    T_h = int(meta.get("history_len", T_total - 1))
    T_p = T_total - T_h

    return CJEPAConfig(
        N=N,
        d=d,
        D=model_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        T_h=T_h,
        T_p=T_p,
        aux_dim=int(aux.shape[-1]),
    )


def infer_episode_ids_from_windows(slots: torch.Tensor, aux: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
    """
    Recover episode ids from sliding-window continuity:
      same episode iff window[i,1:] ~= window[i+1,:-1] (and same for aux)
    """
    n = int(slots.shape[0])
    episode_ids = torch.zeros(n, dtype=torch.long)
    current = 0
    for i in range(n - 1):
        same_slots = torch.allclose(slots[i, 1:], slots[i + 1, :-1], atol=atol, rtol=0.0)
        same_aux = torch.allclose(aux[i, 1:], aux[i + 1, :-1], atol=atol, rtol=0.0)
        if not (same_slots and same_aux):
            current += 1
        episode_ids[i + 1] = current
    return episode_ids


def load_dataset(dataset_path: Path) -> Tuple[PushTTensorDataset, dict, torch.Tensor]:
    payload = torch.load(dataset_path, map_location="cpu")
    slots = payload["slots"].float()
    aux = payload["aux"].float()
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}

    if isinstance(payload, dict) and "episode_ids" in payload:
        episode_ids = payload["episode_ids"].long().view(-1)
    elif isinstance(payload, dict) and "episode_id" in payload:
        episode_ids = payload["episode_id"].long().view(-1)
    else:
        episode_ids = infer_episode_ids_from_windows(slots=slots, aux=aux)

    return PushTTensorDataset(slots=slots, aux=aux), meta, episode_ids


def split_indices_random(n: int, val_ratio: float, seed: int) -> Tuple[list, list]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(n * val_ratio)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def split_indices_episode(episode_ids: torch.Tensor, val_ratio: float, seed: int) -> Tuple[list, list]:
    unique_eps = torch.unique(episode_ids).tolist()
    rng = random.Random(seed)
    rng.shuffle(unique_eps)

    n_val_eps = int(len(unique_eps) * val_ratio)
    n_val_eps = max(1, min(len(unique_eps) - 1, n_val_eps)) if len(unique_eps) > 1 else 1

    val_set = set(unique_eps[:n_val_eps])
    train_idx = [i for i, e in enumerate(episode_ids.tolist()) if e not in val_set]
    val_idx = [i for i, e in enumerate(episode_ids.tolist()) if e in val_set]
    return train_idx, val_idx


def evaluate_future_mse(model: CJEPA, loader: DataLoader, cfg: CJEPAConfig, device: torch.device) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            z = batch["z"].to(device)
            u = batch["u"].to(device)
            B = z.shape[0]

            M = torch.zeros((B, cfg.N), dtype=torch.bool, device=device)
            out = model(z=z, u=u, M=M)

            z_hat = out["z_hat"][:, cfg.T_h :, :, :]
            z_tgt = z[:, cfg.T_h :, :, :]
            mse = (z_hat - z_tgt).pow(2).mean(dim=(1, 2, 3))
            values.extend(mse.detach().cpu().tolist())

    model.train()
    return float(sum(values) / max(1, len(values)))


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train nano C-JEPA on Push-T tensor slots")
    parser.add_argument("--dataset_path", type=str, default=str(default_dataset_path()))
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--exp_name", type=str, default="pusht")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_mode", type=str, default="episode", choices=["random", "episode"])
    parser.add_argument("--mask_min", type=int, default=0)
    parser.add_argument("--mask_max", type=int, default=2)
    parser.add_argument("--model_dim", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--tb_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = Path(args.dataset_path)
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

    dataset, meta, episode_ids = load_dataset(dataset_path)
    if args.split_mode == "episode":
        train_idx, val_idx = split_indices_episode(episode_ids=episode_ids, val_ratio=args.val_ratio, seed=args.seed)
    else:
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

    sample = dataset[0]
    cfg = build_config_from_data(
        slots=sample["z"].unsqueeze(0),
        aux=sample["u"].unsqueeze(0),
        meta=meta,
        model_dim=args.model_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
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
        "dataset_path": str(dataset_path),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "num_episodes": int(torch.unique(episode_ids).numel()),
        "split_mode": args.split_mode,
        "seed": args.seed,
        "tb_dir": str(tb_dir),
        "model": asdict(cfg),
        "train_args": vars(args),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    print(
        f"device={device} split={args.split_mode} episodes={int(torch.unique(episode_ids).numel())} "
        f"train={len(train_idx)} val={len(val_idx)}"
    )

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            z = batch["z"].to(device)
            u = batch["u"].to(device)
            B = z.shape[0]

            M = sample_object_mask(B=B, N=cfg.N, m_min=args.mask_min, m_max=args.mask_max, device=device)
            out = model(z=z, u=u, M=M)
            losses = criterion(z_hat=out["z_hat"], z_target=z, mask_map=out["mask_map"])

            opt.zero_grad(set_to_none=True)
            losses["L_mask"].backward()
            opt.step()
            scheduler.step()

            total_loss += float(losses["L_mask"].detach().cpu())
            n_batches += 1

            writer.add_scalar("train/L_mask", float(losses["L_mask"].detach().cpu()), global_step)
            writer.add_scalar("train/L_history", float(losses["L_history"].detach().cpu()), global_step)
            writer.add_scalar("train/L_future", float(losses["L_future"].detach().cpu()), global_step)
            writer.add_scalar("train/lr", float(opt.param_groups[0]["lr"]), global_step)
            global_step += 1

        train_loss = total_loss / max(1, n_batches)
        val_future_mse = evaluate_future_mse(model, val_loader, cfg, device)
        writer.add_scalar("epoch/train_L_mask", train_loss, epoch)
        writer.add_scalar("epoch/val_future_mse", val_future_mse, epoch)

        print(f"epoch={epoch:03d} train_L_mask={train_loss:.6f} val_future_mse={val_future_mse:.6f}")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": asdict(cfg),
                    "epoch": epoch,
                    "train_L_mask": train_loss,
                    "val_future_mse": val_future_mse,
                    "split_mode": args.split_mode,
                },
                ckpt_path,
            )

    writer.close()


if __name__ == "__main__":
    main()
