import torch
from dataset_sudoku import SudokuDataset

dataset = SudokuDataset(100000)
boards = dataset.data  # shape (100000, 81)
unique_boards = torch.unique(boards, dim=0)

print(f"Total boards: {boards.shape[0]}")
print(f"Unique boards: {unique_boards.shape[0]}")
