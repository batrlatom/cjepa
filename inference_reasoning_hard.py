
import torch
from cjepa.config import ReasoningConfig
from cjepa.model import CJEPA
from cjepa.dataset_reasoning import ReasoningDataset
from cjepa.utils.logging import load_checkpoint
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoint_epoch_49.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ReasoningConfig()
    # Ensure history len matches config used in training (16)
    cfg.history_len = 16 
    cfg.future_len = 16
    
    model = CJEPA(cfg).to(device)
    ckpt_path = os.path.join(args.run_dir, args.checkpoint)
    load_checkpoint(ckpt_path, model)
    model.eval()
    
    dataset = ReasoningDataset(cfg, split="val")
    
    print("\nSTRESS TEST: Max Depth (16)")
    print("-" * 50)
    
    # Manually construct a deep sample
    # Open: ((((((((((((((((
    # Target: ))))))))))))))))
    
    # Or mixed: ([{([{( ...
    
    opens = [0, 2, 4] # ( [ {
    pairs = {0: 1, 2: 3, 4: 5}
    
    # 5 samples
    for i in range(5):
        import random
        stack = []
        open_seq = []
        for _ in range(16):
            op = random.choice(opens)
            open_seq.append(op)
            stack.append(op)
            
        close_seq = [pairs[op] for op in reversed(stack)]
        
        # Embed
        slots_in = [dataset.embedding[t] for t in open_seq]
        slots_out = [dataset.embedding[t] for t in close_seq]
        
        # Combine
        full_seq = slots_in + slots_out
        full_tensor = torch.stack(full_seq).unsqueeze(1).unsqueeze(0).to(device) # (1, 32, 1, D)
        aux = torch.zeros(1, 32, 1).to(device)
        
        # Mask
        mask = torch.zeros((1, 1), dtype=torch.bool, device=device)
        
        with torch.no_grad():
            outputs = model(full_tensor, aux=aux, mask_indices=mask)
            pred = outputs["pred_slots"]
            
            # Decode Future (last 16)
            pred_flat = pred[0, 16:, 0, :]
            pred_str = dataset.decode(pred_flat)
            
            target_str = dataset.decode(torch.stack(slots_out))
            
            is_correct = (pred_str == target_str)
            print(f"Sample {i}: {'✅' if is_correct else '❌'}")
            print(f"Input:  {''.join([dataset.itos[t] for t in open_seq])}")
            print(f"Target: {target_str}")
            print(f"Pred:   {pred_str}")
            print("-" * 20)

if __name__ == "__main__":
    main()
