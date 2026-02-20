import argparse
import numpy as np
import torch
from pathlib import Path

from dataset_sudoku import SudokuDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="sudoku_test_boards.bin")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    # Generate requested number of Valid Sudoku Boards
    dataset = SudokuDataset(size=args.batch_size)
    
    # Shape is (B, 1, 81) -> want (B, 81)
    boards = dataset.data.squeeze(1).numpy().astype(np.int64)
    
    # Generate masks (roughly half cells masked, ~40)
    masks = np.random.rand(args.batch_size, 81) > 0.5
    masks = masks.astype(np.uint8)
    
    print(f"Exporting {args.batch_size} boards to binary...")
    with open(args.output_file, 'wb') as f:
        # Write Boards first (int64)
        f.write(boards.tobytes())
        # Write Masks (uint8)
        f.write(masks.tobytes())
        
    print(f"Saved to {args.output_file} successfully.")
    print("Example board 0:")
    print(boards[0].reshape(9, 9))

if __name__ == "__main__":
    main()
