import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model import CJEPA, CJEPAConfig
from dataset_sudoku import SudokuDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    for p in run_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint_epoch_*.pt in {run_dir}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def split_indices_random(n: int, val_ratio: float, seed: int) -> tuple[list, list]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(n * val_ratio)
    return perm[n_val:], perm[:n_val]


def sample_object_mask(B: int, N: int, k: int, device: torch.device) -> torch.Tensor:
    M = torch.zeros((B, N), dtype=torch.bool, device=device)
    for b in range(B):
        if k > 0:
            indices = torch.randperm(N, device=device)[:k]
            M[b, indices] = True
    return M


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for C-JEPA Sudoku")
    parser.add_argument("--run_dir", type=str, default="runs/sudoku/run_0")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--mask_cells", type=int, default=40, help="Number of cells to hide and predict")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--print_board", action="store_true", help="Print an example board reconstruction")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SudokuDataset(size=args.dataset_size)
    _, val_idx = split_indices_random(len(dataset), val_ratio=args.val_ratio, seed=args.seed)

    if not val_idx:
        raise RuntimeError("Validation split is empty. Increase dataset size or val_ratio.")

    subset = Subset(dataset, val_idx)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    try:
        ckpt_path = resolve_checkpoint(Path(args.run_dir), args.checkpoint)
    except FileNotFoundError as e:
        print(f"Error: {e}. Are you sure you've trained a model first in {args.run_dir}?")
        return
        
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "cfg" in ckpt:
        cfg = CJEPAConfig(**ckpt["cfg"])
    else:
        cfg = CJEPAConfig(
            N=81,
            d=10,
            D=256,
            n_heads=8,
            n_layers=4,
            T_h=1,
            T_p=0,
        )

    model = CJEPA(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    total_acc = 0.0
    n_batches = 0
    total_samples = 0
    
    preview_target = None
    preview_pred = None
    preview_mask = None

    with torch.no_grad():
        for batch in loader:
            z = batch["z"].to(device)
            B = z.shape[0]

            M = sample_object_mask(B, cfg.N, args.mask_cells, device)
            out = model(z=z, M=M)
            z_hat = out["z_hat"]

            preds = z_hat.argmax(dim=-1)
            correct = (preds == z)
            mask_acc = (correct.float() * M.unsqueeze(1).float()).sum() / M.sum().clamp_min(1).float()
            
            total_acc += float(mask_acc.cpu())
            n_batches += 1
            total_samples += B
            
            if args.print_board and preview_target is None:
                preview_target = z[0].cpu().numpy().reshape(9, 9)
                preview_pred = preds[0].cpu().numpy().reshape(9, 9)
                preview_mask = M[0].cpu().numpy().reshape(9, 9)

    overall_acc = total_acc / max(1, n_batches)

    results = {
        "checkpoint": str(ckpt_path),
        "split_mode": "random",
        "num_samples": total_samples,
        "cells_masked": args.mask_cells,
        "mask_accuracy": float(overall_acc)
    }

    print(json.dumps(results, indent=2))
    
    if args.print_board and preview_target is not None:
        print("\n--- Example Board Reconstruction ---\n")
        print(f"{'Input (with masks)':<25} | {'Ground Truth':<25} | {'Predicted':<25}")
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print(f"{'-'*21:<25} | {('-'*21):<25} | {'-'*21}")
            
            in_row = ""
            gt_row = ""
            pr_row = ""
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    in_row += "| "
                    gt_row += "| "
                    pr_row += "| "
                
                is_masked = preview_mask[i, j]
                tgt = preview_target[i, j]
                prd = preview_pred[i, j]
                
                in_row += "_" if is_masked else str(tgt)
                in_row += " "
                
                gt_row += str(tgt)
                gt_row += " "
                
                if is_masked:
                    color = "\033[92m" if prd == tgt else "\033[91m"
                    pr_row += f"{color}{prd}\033[0m "
                else:
                    pr_row += f"{tgt} "
                    
            print(f"{in_row:25} | {gt_row:25} | {pr_row}")
            
        print("\n\033[92mGreen\033[0m = Correct Mask Prediction | \033[91mRed\033[0m = Wrong Mask Prediction | Normal = Unmasked Context")


    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
