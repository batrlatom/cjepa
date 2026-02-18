
import torch
import numpy as np
import argparse
import os
from cjepa.config import DynamicMazeConfig
from cjepa.model import CJEPA
from cjepa.dataset_dynamic_maze import DynamicMazeDataset
from cjepa.utils.logging import load_checkpoint
import collections

def grid_to_actions(grid_tokens):
    size = 15
    grid = grid_tokens.reshape(size, size)
    
    # Init
    start_pos = None
    goal_pos = None
    path_set = set()
    
    for r in range(size):
        for c in range(size):
            val = grid[r, c]
            if val == 2: start_pos = (r, c)
            elif val == 3: goal_pos = (r, c)
            elif val == 4: path_set.add((r, c))
            
    if not start_pos or not goal_pos:
        return None, "Missing S or G"
        
    # BFS on the PREDICTED path pixels to recover order
    # We walk from S to G using only nodes in path_set
    q = collections.deque([(start_pos, [])])
    visited = set([start_pos])
    
    final_actions = None
    
    while q:
        curr, actions = q.popleft()
        if curr == goal_pos:
            final_actions = actions
            break
        
        # Check neighbors
        # We can move to 'Path' (4) or 'Goal' (3)
        r, c = curr
        # Try neighbors
        # Priority? doesn't matter for valid path
        moves = [
            ((-1, 0), 'U'), ((1, 0), 'D'), 
            ((0, -1), 'L'), ((0, 1), 'R')
        ]
        
        for (dr, dc), act in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                if (nr, nc) not in visited:
                    # Valid move if it's in path_set OR it's the goal
                    if (nr, nc) in path_set or (nr, nc) == goal_pos:
                        visited.add((nr, nc))
                        q.append(((nr, nc), actions + [act]))
                        
    if final_actions is not None:
        return final_actions, "Success"
    else:
        return None, "Path Disconnected"

def main():
    parser = argparse.ArgumentParser()
    # Point to the WORKING Dynamic Maze run
    parser.add_argument("--run_dir", type=str, default="runs/dynamic_maze_demo_v4")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_epoch_4.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DynamicMazeConfig()
    model = CJEPA(cfg).to(device)
    
    ckpt_path = os.path.join(args.run_dir, args.checkpoint)
    load_checkpoint(ckpt_path, model)
    model.eval()
    
    dataset = DynamicMazeDataset(cfg, split="val")
    
    print("\nRunning Robust Action Inference (Via Path Parsing)...")
    print("-" * 50)
    
    success_count = 0
    total_count = 5
    
    for i in range(total_count):
        sample = dataset[i]
        
        # 1. Inputs
        slots = sample["slots"].unsqueeze(0).to(device) 
        mask = sample["mask_indices"].unsqueeze(0).to(device)
        aux = sample["aux"].unsqueeze(0).to(device)
        
        # 2. Predict Path (Visual)
        with torch.no_grad():
            outputs = model(slots, aux=aux, mask_indices=mask)
            pred_slots = outputs["pred_slots"]
            
        # 3. Decode
        def decode(slots_tensor):
            dists = torch.cdist(slots_tensor, dataset.embedding.to(device))
            return torch.argmin(dists, dim=1).cpu().numpy()
            
        pred_grid = decode(pred_slots[0, :, 0, :])
        
        # 4. Convert to Actions
        actions, status = grid_to_actions(pred_grid)
        
        print(f"\nSample {i}: {status}")
        
        # Visualization
        # Use existing logic but print extracted actions
        size = 15
        grid = pred_grid.reshape(size, size)
        
        # Colors
        C_RESET = "\033[0m"
        C_WALL = "\033[1;30m" 
        C_START = "\033[1;32m" 
        C_GOAL = "\033[1;31m" 
        C_PATH = "\033[1;33m" 
        
        print("   " + "".join([str(j%10) for j in range(size)]))
        for r in range(size):
            row_str = ""
            for c in range(size):
                val = grid[r, c]
                char = " "
                if val == 0: char = f"{C_WALL}#{C_RESET}"
                elif val == 2: char = f"{C_START}S{C_RESET}"
                elif val == 3: char = f"{C_GOAL}G{C_RESET}"
                elif val == 4: char = f"{C_PATH}.{C_RESET}"
                row_str += char
            print(f"{r:2d} {row_str}")
            
        if actions:
            print(f"Predicted Actions: {actions}")
            success_count += 1
        else:
            print("Could not extract action sequence.")

    print("=" * 50)
    print(f"Action Extraction Success: {success_count}/{total_count}")

if __name__ == "__main__":
    main()
