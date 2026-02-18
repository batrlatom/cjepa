
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import logging

class DynamicMazeDataset(Dataset):
    """
    Dynamic Maze Navigation Task.
    Generates a unique random maze for each sample.
    Goal: Given Walls, Start, Goal (Empty Path), predict the Path.
    """
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        self.size = 15
        self.grid_len = self.size * self.size
        # Virtual length to keep DataLoader happy
        self.length = 10000 if split == "train" else 1000
        
        # Token IDs
        # 0: Wall
        # 1: Empty
        # 2: Start
        # 3: Goal
        # 4: Path
        self.vocab_size = 5
        
        # Fixed Embeddings for the 5 token types
        # We don't embed positions (spatial structure is in the sequence order)
        # But C-JEPA usually takes (T, N, D).
        # We will treat the grid as (225, 1, D) sequence.
        
        g = torch.Generator()
        g.manual_seed(999)
        self.embedding = torch.randn(self.vocab_size, config.slot_dim, generator=g)

    def _generate_maze_and_solve(self):
        size = self.size
        # Walls = 0, Empty = 1
        maze = np.zeros((size, size), dtype=int) 
        
        # Recursive Backtracker for perfect maze
        # 1. Start with all walls
        # 2. Carve
        
        # But we want 0=Wall, 1=Empty
        # Let's start with Walls(0) everywhere
        
        stack = [(1, 1)]
        maze[1, 1] = 1 # Empty
        
        while stack:
            r, c = stack[-1]
            neighbors = []
            
            # Look 2 steps away
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = r + dr, c + dc
                if 0 < nr < size and 0 < nc < size and maze[nr, nc] == 0:
                    neighbors.append((nr, nc))
                    
            if neighbors:
                nr, nc = random.choice(neighbors)
                # Carve path to neighbor
                maze[r + (nr-r)//2, c + (nc-c)//2] = 1
                maze[nr, nc] = 1
                stack.append((nr, nc))
            else:
                stack.pop()
                
        # Pick Start/Goal
        empty_cells = []
        for r in range(size):
            for c in range(size):
                if maze[r, c] == 1:
                    empty_cells.append((r, c))
                    
        if len(empty_cells) < 2: return self._generate_maze_and_solve() # Retry
        
        start = random.choice(empty_cells)
        goal = random.choice(empty_cells)
        while start == goal:
            goal = random.choice(empty_cells)
            
        # BFS Solve
        q = [(start, [])]
        visited = set([start])
        path_cells = []
        
        while q:
            curr, path = q.pop(0)
            if curr == goal:
                path_cells = path + [curr]
                break
            
            r, c = curr
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and maze[nr, nc] == 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), path + [curr]))
                    
        if not path_cells: return self._generate_maze_and_solve() # Retry
        
        # Create Grid with Tokens
        grid = maze.flatten() # 0 or 1
        
        start_idx = start[0]*size + start[1]
        goal_idx = goal[0]*size + goal[1]
        
        grid[start_idx] = 2 # Start
        grid[goal_idx] = 3 # Goal
        
        mask_indices = torch.zeros(self.grid_len, dtype=torch.bool)
        
        # Mark path
        for (r, c) in path_cells:
            if (r, c) != start and (r, c) != goal:
                idx = r*size + c
                grid[idx] = 4 # Path
                mask_indices[idx] = True # We want to mask this 
                
        return grid, mask_indices

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        grid, mask = self._generate_maze_and_solve()
        
        # Convert grid tokens to embeddings
        grid_tensor = torch.tensor(grid, dtype=torch.long)
        slots = self.embedding[grid_tensor] # (225, D)
        slots = slots.unsqueeze(1) # (225, 1, D)
        
        # Mask indices: Reshape to (225, 1)
        mask_indices = mask.unsqueeze(1) # (225, 1)
        
        # Aux is unused but required by model signature
        aux = torch.zeros(self.grid_len, 1) 
        
        # Notes on Masking:
        # In `train.py`, `mask_indices` determines what is MASKED in the input
        # and what loss is calculated on.
        # If we return `mask_indices` here as True for Path tokens,
        # we need logic in `train.py` to use it.
        
        return {
            "slots": slots,      # Ground Truth embeddings (with Path tokens)
            "aux": aux,
            "mask_indices": mask_indices # True = Mask this token
        }
