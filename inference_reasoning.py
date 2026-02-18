
import torch
import os
import argparse
from cjepa.config import ReasoningConfig
from cjepa.model import CJEPA
from cjepa.dataset_reasoning import ReasoningDataset
from cjepa.utils.logging import load_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run directory containing checkpoints")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_epoch_50.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Config & Model
    cfg = ReasoningConfig()
    model = CJEPA(cfg).to(device)
    
    ckpt_path = os.path.join(args.run_dir, args.checkpoint)
    if not os.path.exists(ckpt_path):
        # Try finding latest
        files = os.listdir(args.run_dir)
        ckpts = [f for f in files if f.startswith("checkpoint")]
        if ckpts:
            ckpts.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
            ckpt_path = os.path.join(args.run_dir, ckpts[-1])
            print(f"Using latest checkpoint: {ckpt_path}")
        else:
            print(f"Checkpoint not found at {ckpt_path}")
            return
            
    load_checkpoint(ckpt_path, model)
    model.eval()
    
    # 2. Dataset
    dataset = ReasoningDataset(cfg, split="val")
    # Force specific seed for reproducibility if needed, but validation is random
    
    print("\nRunning Inference on Dyck-2 Task...")
    print("-" * 50)
    
    # 3. Inference Loop
    idxs = range(10) 
    
    correct_count = 0
    total_count = 0
    
    with torch.no_grad():
        for i in idxs:
            sample = dataset[i]
            slots = sample["slots"].unsqueeze(0).to(device) # (1, T, 1, D)
            aux = sample["aux"].unsqueeze(0).to(device)
            
            # Mask future
            B, T, N, D = slots.shape
            mask_indices = torch.zeros((B, N), dtype=torch.bool, device=device)
            
            outputs = model(slots, aux=aux, mask_indices=mask_indices)
            pred_slots = outputs["pred_slots"] # (1, T, 1, D)
            
            # Decode
            # History is first H
            H = cfg.history_len
            
            # Flatten
            input_flat = slots.view(-1, D)
            pred_flat = pred_slots.view(-1, D)
            
            full_str = dataset.decode(input_flat)
            pred_str = dataset.decode(pred_flat)
            
            # Split
            # The sequences are padded.
            # We care about the valid part.
            # The dataset pads with '_' (index 6).
            
            # Input Context involves everything up to H
            # But the dataset construction was: OpenSeq + Padding -> H
            # CloseSeq + Padding -> F
            
            # Actually dataset: full_seq = H + F.
            # Let's inspect the string.
            
            input_context = full_str[:H]
            target_future = full_str[H:]
            pred_future = pred_str[H:]
            
            # Filter padding for metric
            # We want to check if the non-padding characters match.
            def clean(s):
                return s.replace("_", "")
            
            target_clean = clean(target_future)
            pred_clean = clean(pred_future)
            
            is_correct = (target_clean == pred_clean)
            if is_correct: correct_count += 1
            total_count += 1
            
            print(f"\nSample {i}: {'✅' if is_correct else '❌'}")
            print(f"Input:  {input_context.replace('_', ' ')}")
            print(f"Target: {target_future.replace('_', ' ')}")
            print(f"Pred:   {pred_future.replace('_', ' ')}")
            
    print("=" * 50)
    print(f"Accuracy on 10 samples: {correct_count}/{total_count}")

if __name__ == "__main__":
    main()
