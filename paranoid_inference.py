
import torch
import numpy as np
import argparse
import os
from cjepa.config import DynamicMazeConfig
from cjepa.model import CJEPA
from cjepa.dataset_dynamic_maze import DynamicMazeDataset
from cjepa.utils.logging import load_checkpoint

def print_diff(gt_grid, input_grid, pred_grid, mask, title="Maze"):
    size = 15
    display = np.full((size, size*3 + 2), ' ', dtype=str)
    
    # Titles
    # GT | Input (Corrupted) | Pred
    
    def grid_to_char(g):
        d = np.full((size, size), ' ', dtype=str)
        g = g.reshape(size, size)
        for r in range(size):
            for c in range(size):
                val = g[r, c]
                if val == 0: d[r, c] = '#'
                elif val == 1: d[r, c] = ' '
                elif val == 2: d[r, c] = 'S'
                elif val == 3: d[r, c] = 'G'
                elif val == 4: d[r, c] = '.'
        return d

    gt_char = grid_to_char(gt_grid)
    inp_char = grid_to_char(input_grid)
    pred_char = grid_to_char(pred_grid)
    
    print(f"\n--- {title} ---")
    print("GT             | Input (Corrupt)| Prediction")
    for r in range(size):
        row_gt = "".join(gt_char[r])
        row_inp = "".join(inp_char[r])
        row_pred = "".join(pred_char[r])
        print(f"{row_gt} | {row_inp} | {row_pred}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoint_epoch_4.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DynamicMazeConfig()
    model = CJEPA(cfg).to(device)
    
    ckpt_path = os.path.join(args.run_dir, args.checkpoint)
    load_checkpoint(ckpt_path, model)
    model.eval()
    
    dataset = DynamicMazeDataset(cfg, split="val")
    
    print("\nRunning PARANOID Inference (Corrupted Input)...")
    print("Goal: Verify model solves maze WITHOUT seeing the answer in input.")
    print("-" * 50)
    
    success_count = 0
    total_count = 5
    
    for i in range(total_count):
        # 1. Get true sample
        sample = dataset[i]
        
        # 2. Extract Data
        gt_slots = sample["slots"].unsqueeze(0).to(device) # (1, 225, 1, D)
        mask = sample["mask_indices"].unsqueeze(0).to(device) # (1, 225, 1) Bool
        aux = sample["aux"].unsqueeze(0).to(device)
        
        # 3. CORRUPT THE INPUT!
        # We will replace the embeddings at masked positions with the embedding for "Empty" (1)
        # So the model sees "Empty" wherever the "Path" (4) should be.
        # If the model peeks, it will predict "Empty".
        # If it solves, it will predict "Path".
        
        # Create corrupted slots
        empty_token = torch.tensor([1], dtype=torch.long)
        empty_embed = dataset.embedding[empty_token].to(device) # (1, D)
        
        corrupted_slots = gt_slots.clone()
        mask_bool = mask.bool() # (1, 225, 1)
        
        # Apply corruption
        # Expand empty_embed to match shape
        # corrupted_slots[mask_bool] = empty_embed # Shape mismatch potentially
        
        # Do it manually
        B, T, N, D = corrupted_slots.shape
        for t in range(T):
            if mask_bool[0, t, 0]:
                corrupted_slots[0, t, 0, :] = empty_embed[0]
                
        # 4. Feed CORRUPTED slots + Mask to Model
        with torch.no_grad():
            outputs = model(corrupted_slots, aux=aux, mask_indices=mask)
            pred_slots = outputs["pred_slots"]
            
        # 5. Decode
        def decode(slots_tensor):
            dists = torch.cdist(slots_tensor, dataset.embedding.to(device))
            return torch.argmin(dists, dim=1).cpu().numpy()
            
        gt_ids = decode(gt_slots[0, :, 0, :])
        inp_ids = decode(corrupted_slots[0, :, 0, :])
        pred_ids = decode(pred_slots[0, :, 0, :])
        
        # 6. Verify Corruption
        # Path is token 4. Input should NOT have token 4.
        has_path_in_input = np.any(inp_ids == 4)
        if has_path_in_input:
            print("ERROR: Corruption failed! Input still contains Path token.")
            return
            
        # 7. Check Prediction (Reconstruction)
        # We want pred_ids to match gt_ids at masked positions (where inp_ids is 1 but gt_ids is 4)
        
        mask_np = mask[0, :, 0].cpu().numpy()
        pred_correct = np.array_equal(gt_ids[mask_np], pred_ids[mask_np])
        
        print(f"\nSample {i}: {'✅' if pred_correct else '❌'}")
        print_diff(gt_ids, inp_ids, pred_ids, mask_np, title=f"Sample {i}")
        
        if pred_correct: success_count += 1
        
    print("=" * 50)
    print(f"Paranoid Check: {success_count}/{total_count}")
    if success_count == total_count:
        print("VERIFIED: Model solves maze without seeing path tokens.")
    else:
        print("FAILED: Model relies on input leakage?")

if __name__ == "__main__":
    main()
