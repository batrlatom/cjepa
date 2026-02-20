import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from model import CJEPA, CJEPAConfig


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


def resolve_checkpoint(run_dir: Path, checkpoint: str | None) -> Path:
    if checkpoint is not None:
        p = Path(checkpoint)
        if p.exists():
            return p
        p2 = run_dir / checkpoint
        if p2.exists():
            return p2
        raise FileNotFoundError(f"checkpoint not found: {checkpoint} or {p2}")

    pattern = re.compile(r"checkpoint_epoch_(\d+)\.pt$")
    candidates = []
    for p in run_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint_epoch_*.pt in {run_dir}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def infer_episode_ids_from_windows(slots: torch.Tensor, aux: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
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


def split_indices_random(n: int, val_ratio: float, seed: int) -> tuple[list, list]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(n * val_ratio)
    return perm[n_val:], perm[:n_val]


def split_indices_episode(episode_ids: torch.Tensor, val_ratio: float, seed: int) -> tuple[list, list]:
    unique_eps = torch.unique(episode_ids).tolist()
    rng = random.Random(seed)
    rng.shuffle(unique_eps)

    n_val_eps = int(len(unique_eps) * val_ratio)
    n_val_eps = max(1, min(len(unique_eps) - 1, n_val_eps)) if len(unique_eps) > 1 else 1

    val_set = set(unique_eps[:n_val_eps])
    train_idx = [i for i, e in enumerate(episode_ids.tolist()) if e not in val_set]
    val_idx = [i for i, e in enumerate(episode_ids.tolist()) if e in val_set]
    return train_idx, val_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for nano C-JEPA Push-T")
    parser.add_argument("--run_dir", type=str, default="runs/pusht/run_0")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=str(default_dataset_path()))
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_mode", type=str, default="episode", choices=["random", "episode"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.dataset_path, map_location="cpu")
    slots = payload["slots"].float()
    aux = payload["aux"].float()
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    dataset = PushTTensorDataset(slots=slots, aux=aux)

    if isinstance(payload, dict) and "episode_ids" in payload:
        episode_ids = payload["episode_ids"].long().view(-1)
    elif isinstance(payload, dict) and "episode_id" in payload:
        episode_ids = payload["episode_id"].long().view(-1)
    else:
        episode_ids = infer_episode_ids_from_windows(slots=slots, aux=aux)

    if args.split_mode == "episode":
        _, val_idx = split_indices_episode(episode_ids=episode_ids, val_ratio=args.val_ratio, seed=args.seed)
    else:
        _, val_idx = split_indices_random(len(dataset), val_ratio=args.val_ratio, seed=args.seed)

    if not val_idx:
        raise RuntimeError("Validation split is empty. Increase dataset size or val_ratio.")

    n = min(args.num_samples, len(val_idx))
    subset = Subset(dataset, val_idx[:n])
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt_path = resolve_checkpoint(Path(args.run_dir), args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "cfg" in ckpt:
        cfg = CJEPAConfig(**ckpt["cfg"])
    else:
        T_total = slots.shape[1]
        T_h = int(meta.get("history_len", T_total - 1))
        cfg = CJEPAConfig(
            N=int(slots.shape[2]),
            d=int(slots.shape[3]),
            D=128,
            n_heads=8,
            n_layers=6,
            T_h=T_h,
            T_p=int(T_total - T_h),
            aux_dim=int(aux.shape[-1]),
        )

    model = CJEPA(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mse_vals = []
    cos_vals = []
    l2_vals = []
    previews = []

    with torch.no_grad():
        for batch in loader:
            z = batch["z"].to(device)
            u = batch["u"].to(device)
            B = z.shape[0]

            M = torch.zeros((B, cfg.N), dtype=torch.bool, device=device)
            out = model(z=z, u=u, M=M)

            z_hat_future = out["z_hat"][:, cfg.T_h :, :, :]
            z_true_future = z[:, cfg.T_h :, :, :]

            diff = z_hat_future - z_true_future
            mse = diff.pow(2).mean(dim=(1, 2, 3))
            l2 = diff.pow(2).sum(dim=(1, 2, 3)).sqrt()

            z_hat_flat = z_hat_future.flatten(start_dim=1)
            z_true_flat = z_true_future.flatten(start_dim=1)
            cos = torch.nn.functional.cosine_similarity(z_hat_flat, z_true_flat, dim=1)

            mse_vals.extend(mse.detach().cpu().tolist())
            l2_vals.extend(l2.detach().cpu().tolist())
            cos_vals.extend(cos.detach().cpu().tolist())

            if len(previews) < 3:
                take = min(3 - len(previews), B)
                for i in range(take):
                    previews.append(
                        {
                            "target_t0_slot0": z_true_future[i, 0, 0, :8].detach().cpu().tolist(),
                            "pred_t0_slot0": z_hat_future[i, 0, 0, :8].detach().cpu().tolist(),
                        }
                    )

    results = {
        "checkpoint": str(ckpt_path),
        "dataset_path": str(args.dataset_path),
        "split_mode": args.split_mode,
        "num_episodes": int(torch.unique(episode_ids).numel()),
        "num_samples": len(mse_vals),
        "future_mse_mean": float(sum(mse_vals) / max(1, len(mse_vals))),
        "future_cosine_mean": float(sum(cos_vals) / max(1, len(cos_vals))),
        "future_l2_mean": float(sum(l2_vals) / max(1, len(l2_vals))),
        "success_rate_future_mse_le_1e-4": float(sum(v <= 1e-4 for v in mse_vals) / max(1, len(mse_vals))),
        "success_rate_future_mse_le_1e-3": float(sum(v <= 1e-3 for v in mse_vals) / max(1, len(mse_vals))),
        "sample_previews": previews,
    }

    print(json.dumps(results, indent=2))

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
