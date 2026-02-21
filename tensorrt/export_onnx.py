import argparse
import pickle
import re
from pathlib import Path

# Allow direct script execution: python tensorrt/export_onnx.py ...
if __package__ in (None, ""):
    import os as _os
    import sys as _sys

    _REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    if _REPO_ROOT not in _sys.path:
        _sys.path.insert(0, _REPO_ROOT)

import torch

from model import CJEPA, CJEPAConfig
from vision import VisionBackboneWrapper


class RobomimicExportWrapper(torch.nn.Module):
    """Export wrapper for the Robomimic policy + vision encoder."""

    def __init__(self, policy: CJEPA, vision_encoder: VisionBackboneWrapper, vision_dim: int):
        super().__init__()
        self.policy = policy
        self.vision_encoder = vision_encoder
        self.vision_dim = int(vision_dim)

    def forward(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        # images: (B,T,C,H,W) or (B,T,V,C,H,W)
        # proprio: (B,T,proprio_dim)
        bsz, t_obs = images.shape[:2]

        if images.ndim == 5:
            imgs_flat = images.reshape(bsz * t_obs, *images.shape[2:])
            vision_feats_flat = self.vision_encoder(imgs_flat, return_spatial=False)
            vision_feats = vision_feats_flat.view(bsz, t_obs, self.vision_dim)
        elif images.ndim == 6:
            n_views = images.shape[2]
            imgs_flat = images.reshape(bsz * t_obs * n_views, *images.shape[3:])
            vision_feats_flat = self.vision_encoder(imgs_flat, return_spatial=False)
            vision_feats = vision_feats_flat.view(bsz, t_obs, n_views, self.vision_dim).mean(dim=2)
        else:
            raise ValueError(f"Unexpected image tensor rank: {images.ndim}")

        ctx = torch.cat([vision_feats, proprio], dim=-1)
        ctx_latents = self.policy.context_norm(ctx)
        return self.policy(ctx_latents)  # (B, pred_horizon, action_dim)


def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        print("Checkpoint has non-tensor metadata. Retrying with weights_only=False...")
        return torch.load(path, map_location=map_location, weights_only=False)


def resolve_checkpoint(run_dir: Path, checkpoint: str | None) -> Path:
    if checkpoint is not None:
        p = Path(checkpoint)
        if p.exists():
            return p
        p2 = run_dir / checkpoint
        if p2.exists():
            return p2
        raise FileNotFoundError(f"checkpoint not found: {checkpoint} or {p2}")

    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    # Preferred current checkpoint name for robomimic training.
    best = run_dir / "best_model.pt"
    if best.exists():
        return best

    # Fallback for legacy naming patterns.
    pattern = re.compile(r"checkpoint_epoch_(\d+)\.pt$")
    candidates = []
    for p in run_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    # Last fallback: most recently modified *.pt
    pt_files = list(run_dir.glob("*.pt"))
    if pt_files:
        return max(pt_files, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(f"no checkpoint file found in {run_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="runs/robomimic/run_1")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--num_cameras", type=int, default=2)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpt_path = resolve_checkpoint(run_dir, args.checkpoint).resolve()
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = _safe_torch_load(ckpt_path, map_location="cpu")

    expected = {"policy_state", "vision_state", "cfg"}
    if not expected.issubset(ckpt.keys()):
        raise KeyError(
            f"Checkpoint at {ckpt_path} is not a robomimic policy checkpoint. "
            f"Expected keys {sorted(expected)}, got keys {sorted(ckpt.keys())}."
        )

    cfg = CJEPAConfig(**ckpt["cfg"])
    policy = CJEPA(cfg)
    vision_encoder = VisionBackboneWrapper(pretrained=False)
    policy.load_state_dict(ckpt["policy_state"])
    vision_encoder.load_state_dict(ckpt["vision_state"])
    policy.eval()
    vision_encoder.eval()

    wrapped_model = RobomimicExportWrapper(policy, vision_encoder, vision_dim=cfg.vision_dim)
    wrapped_model.eval()

    bsz = int(args.batch_size)
    t_obs = int(cfg.obs_horizon)
    if int(args.num_cameras) > 1:
        images = torch.randn((bsz, t_obs, int(args.num_cameras), 3, args.image_size, args.image_size), dtype=torch.float32)
    else:
        images = torch.randn((bsz, t_obs, 3, args.image_size, args.image_size), dtype=torch.float32)
    proprio = torch.randn((bsz, t_obs, cfg.proprio_dim), dtype=torch.float32)

    onnx_path = run_dir / f"model_b{bsz}.onnx"
    torch.onnx.export(
        wrapped_model,
        (images, proprio),
        str(onnx_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["images", "proprio"],
        output_names=["pred_actions"],
    )
    print(f"Exported robomimic ONNX to {onnx_path}")


if __name__ == "__main__":
    main()
