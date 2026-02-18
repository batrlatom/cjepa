
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import logging

class MazeDataset(Dataset):
    """
    Fixed Maze Navigation Task.
    The model must learn the structure of a static 15x15 maze and predict the shortest path
    from any Start to any Goal.
    
    Input: Slot 0 = Current Pos, Aux 0 = Goal Pos
    Target: Next Pos
    """
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        self.size = 15
        self.num_samples = 10000 if split == "train" else 1000
        
        # 1. Generate Fixed Maze (Seed 42)
        self.maze, self.graph = self._generate_maze(self.size, seed=42)
        
        # 2. Pre-compute/Cache paths? 
        # On-the-fly BFS is fast enough for 15x15.
        
        # Embeddings
        # Grid positions: size*size
        self.vocab_size = self.size * self.size
        # Valid positions are only open spaces
        self.valid_positions = [r*self.size + c for r in range(self.size) for c in range(self.size) if self.maze[r,c] == 0]
        
        # Fixed random embeddings
        g = torch.Generator()
        g.manual_seed(999)
        self.embedding = torch.randn(self.vocab_size, config.slot_dim, generator=g)

    def _generate_maze(self, size, seed):
        np.random.seed(seed)
        # Prim's Algorithm
        maze = np.ones((size, size), dtype=int)
        
        # Start at (1,1)
        maze[1, 1] = 0
        walls = [(1, 2), (2, 1)]
        
        while walls:
            # Pick random wall
            idx = np.random.randint(len(walls))
            r, c = walls[idx]
            walls.pop(idx)
            
            # Check neighbors
            neighbors = []
            if r > 1 and maze[r-2, c] == 0: neighbors.append((r-2, c))
            if r < size-2 and maze[r+2, c] == 0: neighbors.append((r+2, c))
            if c > 1 and maze[r, c-2] == 0: neighbors.append((r, c-2))
            if c < size-2 and maze[r, c+2] == 0: neighbors.append((r, c+2))
            
            # If exactly one visited neighbor is 2 steps away?
            # Simplified Prim's: neighbors 2 steps away
            
            # Actually simple DFS is easier to implement robustly
            pass
        
        # Let's use DFS recursive
        maze = np.ones((size, size), dtype=int)
        
        def carve(r, c):
            maze[r, c] = 0
            dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            np.random.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 1 <= nr < size-1 and 1 <= nc < size-1 and maze[nr, nc] == 1:
                    maze[r + dr//2, c + dc//2] = 0
                    carve(nr, nc)
                    
        carve(1, 1)
        return maze, None

    def _solve_bfs(self, start, end):
        # start, end are indices
        sr, sc = start // self.size, start % self.size
        er, ec = end // self.size, end % self.size
        
        q = [(sr, sc, [])]
        visited = set([(sr, sc)])
        
        while q:
            r, c, path = q.pop(0)
            if r == er and c == ec:
                return path + [(r, c)]
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and self.maze[nr, nc] == 0:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc, path + [(r, c)]))
        return []

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Pick random Start/End
        while True:
            s_idx = random.choice(self.valid_positions)
            e_idx = random.choice(self.valid_positions)
            if s_idx != e_idx:
                path = self._solve_bfs(s_idx, e_idx)
                if len(path) > 1:
                    break
        
        # Path is list of (r, c)
        # Convert to flat indices
        path_indices = [r*self.size + c for r, c in path]
        
        # We need fixed length T
        # T = history + future
        # Let's say T = 16 (Config)
        # If path is shorter, we pad
        # If path is longer, we crop a random segment
        
        total_len = self.config.history_len + self.config.future_len
        
        if len(path_indices) > total_len:
            # Crop
            start = random.randint(0, len(path_indices) - total_len)
            seq = path_indices[start : start + total_len]
        else:
            # Pad at end with goal
            seq = path_indices
            while len(seq) < total_len:
                seq.append(e_idx) # Pad with staying at goal
                
        # Embed
        # Input: The sequence
        # Goal: e_idx
        
        slots = self.embedding[seq] # (T, D)
        slots = slots.unsqueeze(1) # (T, 1, D)
        
        # Aux: Goal embedding
        goal_vec = self.embedding[e_idx] # (D)
        aux = goal_vec.view(1, -1).expand(total_len, -1) # (T, D)
        # Aux dim might need to be embedding dim?
        # In config we might set aux_dim = slot_dim
        
        return {
            "slots": slots,
            "aux": aux
        }
