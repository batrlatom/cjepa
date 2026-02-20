import argparse
import re
from pathlib import Path
import torch

from model import CJEPA, CJEPAConfig

class CJEPAExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z, M):
        return self.model(z=z, M=M)["z_hat"]

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="runs/sudoku/run_3")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = resolve_checkpoint(run_dir, args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "cfg" in ckpt:
        cfg = CJEPAConfig(**ckpt["cfg"])
    else:
        cfg = CJEPAConfig(N=81, d=10, D=256, n_heads=8, n_layers=4, T_h=1, T_p=0)

    model = CJEPA(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapped_model = CJEPAExportWrapper(model)

    B = args.batch_size
    z = torch.ones((B, 1, cfg.N), dtype=torch.long)
    M = torch.zeros((B, cfg.N), dtype=torch.bool)

    onnx_path = run_dir / f"model_b{B}.onnx"
    torch.onnx.export(
        wrapped_model,
        (z, M),
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['z', 'M'],
        output_names=['z_hat']
    )
    print(f"Exported to {onnx_path}")

if __name__ == "__main__":
    main()
