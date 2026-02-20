import random
import torch
from torch.utils.data import Dataset

def generate_base_board():
    # A valid 9x9 sudoku board
    base = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [4, 5, 6, 7, 8, 9, 1, 2, 3],
        [7, 8, 9, 1, 2, 3, 4, 5, 6],
        [2, 3, 1, 5, 6, 4, 8, 9, 7],
        [5, 6, 4, 8, 9, 7, 2, 3, 1],
        [8, 9, 7, 2, 3, 1, 5, 6, 4],
        [3, 1, 2, 6, 4, 5, 9, 7, 8],
        [6, 4, 5, 9, 7, 8, 3, 1, 2],
        [9, 7, 8, 3, 1, 2, 6, 4, 5]
    ]
    return torch.tensor(base, dtype=torch.long)

def shuffle_board(board):
    # 1. Shuffle numbers 1-9
    nums = list(range(1, 10))
    random.shuffle(nums)
    mapping = {i: nums[i-1] for i in range(1, 10)}
    mapping[0] = 0
    new_board = board.clone()
    for i in range(1, 10):
        new_board[board == i] = mapping[i]
    
    # 2. Shuffle rows within blocks
    for i in range(0, 9, 3):
        rows = list(range(i, i+3))
        random.shuffle(rows)
        new_board[i:i+3] = new_board[rows]
        
    # 3. Shuffle cols within blocks
    for i in range(0, 9, 3):
        cols = list(range(i, i+3))
        random.shuffle(cols)
        new_board[:, i:i+3] = new_board[:, cols]
        
    # 4. Shuffle 3x3 blocks row-wise
    blocks = [0, 1, 2]
    random.shuffle(blocks)
    temp = new_board.clone()
    for i in range(3):
        new_board[i*3:(i+1)*3] = temp[blocks[i]*3:(blocks[i]+1)*3]
        
    # 5. Shuffle 3x3 blocks col-wise
    blocks = [0, 1, 2]
    random.shuffle(blocks)
    temp = new_board.clone()
    for i in range(3):
        new_board[:, i*3:(i+1)*3] = temp[:, blocks[i]*3:(blocks[i]+1)*3]
        
    return new_board

class SudokuDataset(Dataset):
    def __init__(self, size: int):
        """
        size: number of boards to generate and cache.
        """
        self.size = size
        base = generate_base_board()
        
        # Precompute all boards to avoid on-the-fly generation slowdowns
        print(f"Precomputing {size} Sudoku boards...")
        boards = []
        for _ in range(size):
            board = shuffle_board(base).view(1, 81)
            boards.append(board)
        
        self.data = torch.cat(boards, dim=0)
        print("Done precomputing.")
        
    def __len__(self) -> int:
        return self.size
        
    def __getitem__(self, idx: int):
        return {
            "z": self.data[idx:idx+1],  # Maintain (T, N) shape -> (1, 81)
            "u": torch.zeros(0)  # No aux variables for Sudoku
        }
