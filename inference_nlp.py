
import torch
import os
import argparse
from cjepa.config import NLPConfig
from cjepa.model import CJEPA
from cjepa.dataset_nlp import NLPDataset
from cjepa.utils.logging import load_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run directory containing checkpoints")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_epoch_0.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Config & Model
    cfg = NLPConfig()
    model = CJEPA(cfg).to(device)
    
    ckpt_path = os.path.join(args.run_dir, args.checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return
        
    epoch = load_checkpoint(ckpt_path, model)
    model.eval()
    
    # 2. Dataset (to get embeddings and vocab)
    # We need the exact same embeddings as training
    dataset = NLPDataset(cfg, split="val", download=True)
    embeddings = dataset.embedding.to(device) # (Vocab, D)
    
    # Vocab mapping
    itos = dataset.itos
    
    print("\nRunning Inference...")
    print("-" * 50)
    
    # 3. Inference Loop
    # Take a few samples
    idxs = [0, 100, 500, 1000] # Arbitrary indices
    
    model.eval()
    with torch.no_grad():
        for i in idxs:
            if i >= len(dataset): break
            
            sample = dataset[i]
            slots = sample["slots"].unsqueeze(0).to(device) # (1, T, N, D)
            aux = sample["aux"].unsqueeze(0).to(device)
            
            # Forward
            # For inference: we want to predict FUTURE.
            # The model is trained with masking.
            # To "generate", we should mask the future tokens and see what it fills in.
            
            # Mask generation:
            # We want to mask ALL future tokens (t >= history_len)
            B, T, N, D = slots.shape
            history_len = cfg.history_len
            
            # Construct manual mask
            # Mask indices: (B, N) -> This controls "object masking" in history.
            # But the model *also* applies future masking internally based on indices.
            # If we pass mask_indices=None, no history objects are masked.
            # But future (t >= history_len) is ALWAYS masked in loss calculation...
            # Wait, `model.py` applies masking logic:
            # "Future entities are always masked for prediction" - typically implied by the task.
            # In my `model.py`, I implemented explicit masking logic triggered by `mask_indices`.
            # Let's check `model.py` logic again.
            
            # Logic in model.py:
            # if mask_indices is not None:
            #   Refine logic:
            #   is_future = time_indices >= self.cfg.history_len
            #   is_history_mask = ...
            #   final_mask = is_future | is_history_mask
            #   Replace masked slots with mask tokens.
            
            # So if I pass mask_indices = False (all zeros), then `is_history_mask` is False.
            # But `is_future` will be True for t >= history_len.
            # So the model WILL replace future slots with mask tokens if I pass *any* mask_indices tensor.
            # If mask_indices is None, it skips the masking block entirely!
            
            # Correction: Implementation in `model.py` lines 113+:
            # `if mask_indices is not None:` -> proceeds to mask.
            # If None, it does NOT mask anything (returns raw embeddings).
            
            # So for inference, I MUST provide a dummy mask_indices tensor of all False
            # to trigger the "Future Masking" logic inside the block.
            
            mask_indices = torch.zeros((B, N), dtype=torch.bool, device=device)
            
            outputs = model(slots, aux=aux, mask_indices=mask_indices)
            pred_slots = outputs["pred_slots"] # (1, T, N, D)
            
            # 4. Decode
            # We compare Input vs Prediction
            
            # Flatten to chars
            # (1, T, N, D) -> (T*N, D)
            target_flat = slots.view(-1, D)
            pred_flat = pred_slots.view(-1, D)
            
            # Nearest Neighbor
            # Dist: (L, 1, D) - (1, V, D) -> (L, V, D) -> norm -> (L, V)
            # Or cosine sim
            
            def decode(vectors):
                # vectors: (M, D)
                # embeddings: (V, D)
                dists = torch.cdist(vectors, embeddings) # (M, V)
                ids = torch.argmin(dists, dim=1)
                chars = "".join([itos[idx.item()] for idx in ids])
                return chars
            
            # Split into History (Context) and Future (Pred)
            # Reshape text to lines of length N
            
            input_text = decode(target_flat)
            pred_text = decode(pred_flat)
            
            # Format
            # Chunk is (T, N).
            # T = H + F.
            T_total = cfg.history_len + cfg.future_len
            H = cfg.history_len
            N_slots = cfg.num_slots
            
            # The text was reshaped to (T, N).
            # Let's verify how it reads.
            # "Row by row" means T=0 is first N chars, T=1 is next N chars.
            
            # We want to show:
            # Context: The first H*N chars.
            # Target Future: The last F*N chars.
            # Pred Future: The last F*N chars from prediction.
            
            split_idx = H * N_slots
            
            context_str = input_text[:split_idx]
            target_fut_str = input_text[split_idx:]
            pred_fut_str = pred_text[split_idx:]
            
            # Insert newlines for readability (every N chars)
            def format_block(s, width):
                visible = s.replace("\n", "¶").replace(" ", "·")
                lines = [visible[k:k+width] for k in range(0, len(visible), width)]
                return "\n".join(lines)
                
            print(f"\n--- Sample {i} ---")
            print("CONTEXT:")
            print(format_block(context_str, N_slots))
            print("-" * 20)
            print("TARGET FUTURE:")
            print(format_block(target_fut_str, N_slots))
            print("-" * 20)
            print("PREDICTED FUTURE:")
            print(format_block(pred_fut_str, N_slots))
            print("=" * 50)

if __name__ == "__main__":
    main()
