
import torch
import numpy as np
import argparse
import os
from cjepa.config import MazeConfig
from cjepa.model import CJEPA
from cjepa.dataset_maze import MazeDataset
from cjepa.utils.logging import load_checkpoint

def print_maze_path(maze_grid, path_indices, start_idx, goal_idx, title="Path"):
    """
    Visualizes the path on the maze.
    maze_grid: 15x15 np array (1=wall, 0=empty)
    path_indices: list of flat indices
    """
    size = maze_grid.shape[0]
    display = np.full((size, size), ' ', dtype=str)
    
    # Walls
    display[maze_grid == 1] = '#'
    
    # Path
    for idx in path_indices:
        r, c = idx // size, idx % size
        display[r, c] = '.'
        
    # Start/Goal
    sr, sc = start_idx // size, start_idx % size
    gr, gc = goal_idx // size, goal_idx % size
    
    display[sr, sc] = 'S'
    display[gr, gc] = 'G'
    
    print(f"\n--- {title} ---")
    print("   " + "".join([str(i%10) for i in range(size)]))
    for r in range(size):
        row_str = "".join(display[r])
        print(f"{r:2d} {row_str}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoint_epoch_99.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MazeConfig()
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
    
    # Dataset holds the fixed maze
    dataset = MazeDataset(cfg, split="val")
    maze_grid = dataset.maze
    
    print("\nRunning Maze Inference...")
    print("-" * 50)
    
    # Test on 5 samples
    # We want to check: Given History (Partial Path), does it predict Future (Rest of Path)?
    
    # Actually, the dataset serves:
    # Slots: [History... Future...]
    # Aux: [Goal... Goal...]
    # We feed [History] and ask for [Future].
    
    H = cfg.history_len
    total_len = H + cfg.future_len
    
    success_count = 0
    total_count = 0
    
    idxs = range(5)
    
    with torch.no_grad():
        for i in idxs:
            sample = dataset[i]
            slots = sample["slots"].unsqueeze(0).to(device) # (1, T, 1, D)
            aux = sample["aux"].unsqueeze(0).to(device)
            
            # Extract Goal Index from Aux
            # Recall aux is embedding of goal index
            # We can't easily reverse lookup embedding to index unless we brute force or store it
            # But we can infer it from the dataset logic or valid_positions
            
            # Wait, for visualization we need the actual indices.
            # Convert input slots back to indices
            def vec_to_idx(vec):
                dists = torch.cdist(vec, dataset.embedding.to(vec.device))
                return torch.argmin(dists, dim=1)
                
            input_indices = vec_to_idx(slots[0, :, 0, :]).cpu().numpy()
            
            # Goal is aux[0]
            goal_idx = vec_to_idx(aux[0, 0:1, :]).item()
            start_idx = input_indices[0]
            
            # Mask
            mask = torch.zeros((1, 1), dtype=torch.bool, device=device)
            
            outputs = model(slots, aux=aux, mask_indices=mask)
            pred_slots = outputs["pred_slots"] # (1, T, 1, D)
            
            pred_indices = vec_to_idx(pred_slots[0, :, 0, :]).cpu().numpy()
            
            # Metrics
            # Target Future
            target_future = input_indices[H:]
            pred_future = pred_indices[H:]
            
            # Did we reach the goal?
            # Or did we match the path?
            # If we predict *a* valid path it's good, but we trained on *shortest* path.
            # So let's check exact match.
            
            match = np.array_equal(target_future, pred_future)
            if match: success_count += 1
            total_count += 1
            
            print(f"\nSample {i}: {'✅' if match else '❌'}")
            
            # Visualize Full Path (History + Predicted Future)
            full_pred_path = np.concatenate([input_indices[:H], pred_future])
            print_maze_path(maze_grid, full_pred_path, start_idx, goal_idx, title=f"Sample {i}: Pred")
            
            # Visualize Target
            # print_maze_path(maze_grid, input_indices, start_idx, goal_idx, title=f"Sample {i}: Target")

    print("=" * 50)
    print(f"Accuracy (Exact Path Match): {success_count}/{total_count}")

if __name__ == "__main__":
    main()
