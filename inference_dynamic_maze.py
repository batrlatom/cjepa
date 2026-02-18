
import torch
import numpy as np
import argparse
import os
from cjepa.config import DynamicMazeConfig
from cjepa.model import CJEPA
from cjepa.dataset_dynamic_maze import DynamicMazeDataset
from cjepa.utils.logging import load_checkpoint

def print_dynamic_maze(grid_tokens, title="Maze"):
    """
    Visualizes the maze.
    grid_tokens: (225,) array of token IDs
    0: Wall, 1: Empty, 2: Start, 3: Goal, 4: Path
    """
    size = 15 
    grid = grid_tokens.reshape(size, size)
    
    # Colors
    C_RESET = "\033[0m"
    C_WALL = "\033[1;30m" # Bright Black/Gray
    C_START = "\033[1;32m" # Bright Green
    C_GOAL = "\033[1;31m" # Bright Red
    C_PATH = "\033[1;33m" # Bright Yellow
    C_EMPTY = " "
    
    display = np.full((size, size), ' ', dtype=object) # Use object to prevent string truncation
    
    for r in range(size):
        for c in range(size):
            val = grid[r, c]
            if val == 0: display[r, c] = f"{C_WALL}#{C_RESET}"
            elif val == 1: display[r, c] = f" " # Just space
            elif val == 2: display[r, c] = f"{C_START}S{C_RESET}"
            elif val == 3: display[r, c] = f"{C_GOAL}G{C_RESET}"
            elif val == 4: display[r, c] = f"{C_PATH}.{C_RESET}"
            
    print(f"\n--- {title} ---")
    print("   " + "".join([str(i%10) for i in range(size)]))
    for r in range(size):
        row_str = "".join(display[r])
        print(f"{r:2d} {row_str}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoint_epoch_49.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DynamicMazeConfig()
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
            
    load_checkpoint(ckpt_path, model)
    model.eval()
    
    # Dataset just for embeddings reuse
    dataset = DynamicMazeDataset(cfg, split="val")
    
    print("\nRunning Dynamic Maze Inference...")
    print("-" * 50)
    
    success_count = 0
    total_count = 0
    
    # Run 5 samples
    for i in range(5):
        # Generate fresh sample
        sample = dataset[i] # This Triggers generation!
        
        # Ground Truth Slots (Contains Path)
        gt_slots = sample["slots"].unsqueeze(0).to(device) # (1, 225, 1, D)
        mask = sample["mask_indices"].unsqueeze(0).to(device) # (1, 225, 1)
        aux = sample["aux"].unsqueeze(0).to(device)
        
        # To simulate the task, we need to feed the Model:
        # The Slots, but masked!
        # wait, C-JEPA `forward` takes `mask_indices`.
        # Inside `forward`:
        # 1. Embed `slots`
        # 2. Replaces slots where mask==True with MASK_TOKEN.
        # 3. Predicts unmasked (reconstructed) slots.
        
        # So we just feed gt_slots and mask.
        
        with torch.no_grad():
            outputs = model(gt_slots, aux=aux, mask_indices=mask)
            pred_slots = outputs["pred_slots"] # (1, 225, 1, D)
            
        # Decode
        # We need to map embeddings back to tokens.
        # Dataset embeddings: 5 tokens
        # We can implement nearest neighbor.
        
        def decode(slots_tensor):
            # slots_tensor: (225, D)
            # embeddings: (5, D)
            dists = torch.cdist(slots_tensor, dataset.embedding.to(device))
            ids = torch.argmin(dists, dim=1)
            return ids.cpu().numpy()
            
        gt_ids = decode(gt_slots[0, :, 0, :])
        pred_ids = decode(pred_slots[0, :, 0, :])
        
        # We only care about the masked positions (Path)
        # But let's verify everything.
        
        # Construct Reconstruction Grid
        # Where mask is false, use GT (Input)
        # Where mask is true, use Pred
        mask_np = mask[0, :, 0].cpu().numpy()
        
        recon_ids = gt_ids.copy()
        recon_ids[mask_np] = pred_ids[mask_np]
        
        # Compare Path Quality
        # Correct if all masked/path tokens are predicted as Path(4)?
        # Or exact match?
        
        # Actually in this tokenization:
        # Start(2), Goal(3), Wall(0), Empty(1), Path(4).
        # The dataset sets Path(4) for the shortest path.
        # If the model predicts Path(4) exactly where GT has Path(4), it's correct.
        
        # Check mismatches on path
        path_mask = (gt_ids == 4)
        pred_path_correct = np.array_equal(gt_ids[path_mask], pred_ids[path_mask])
        
        # Also check no "hallucination" (predicting path elsewhere)
        # Ideally recon should match gt exactly
        exact_match = np.array_equal(gt_ids, pred_ids) # C-JEPA reconstructs ALL tokens
        
        print(f"\nSample {i}: {'✅' if exact_match else '❌'}")
        
        # print_dynamic_maze(gt_ids, title="Ground Truth")
        print_dynamic_maze(recon_ids, title="Predicted")
        
        if exact_match:
            success_count += 1
        total_count += 1
        
    print("=" * 50)
    print(f"Accuracy: {success_count}/{total_count}")

if __name__ == "__main__":
    main()
