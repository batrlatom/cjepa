
import torch
import torch.nn as nn
import torch.optim as optim
from cjepa.config import NLPConfig
from cjepa.model import CJEPA
from cjepa.loss import CJEPALoss
from torch.utils.data import DataLoader, Dataset

# Configuration
cfg = NLPConfig()
cfg.dropout = 0.0 # Disable dropout for overfitting
cfg.history_len = 5
cfg.future_len = 5
cfg.num_slots = 1 # 1 char per slot for simplicity
cfg.model_dim = 128
cfg.num_layers = 2
cfg.slot_dim = 32

# Data: "hello world" repeated
text = "hello world " * 10
chars = sorted(list(set(text)))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

# Fixed embeddings
embedding = torch.randn(len(chars), cfg.slot_dim)

class SingleBatchDataset(Dataset):
    def __init__(self):
        self.data = [text] # Just one string
    def __len__(self):
        return 100 # Artificially large
    def __getitem__(self, idx):
        # Always return "hello world " split into history/future
        # "hello" (5) -> " world" (5) (approx)
        full_str = "hello worel " # 12 chars. 
        # T = 5+5 = 10.
        # Let's use "0123456789"
        s = "0123456789"
        ids = torch.tensor([int(c) for c in s], dtype=torch.long)
        
        # Actually use "hello worl" (10 chars)
        s = "hello worl"
        ids = torch.tensor([stoi[c] for c in s], dtype=torch.long)
        
        slots = embedding[ids] # (10, 32)
        # Reshape to (T, N, D) -> (10, 1, 32)
        slots = slots.unsqueeze(1)
        
        return {
            "slots": slots,
            "aux": torch.zeros(10, 1) # Dummy aux
        }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CJEPA(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = CJEPALoss(history_len=cfg.history_len)
    
    dataset = SingleBatchDataset()
    loader = DataLoader(dataset, batch_size=8)
    
    print("Starting Overfit Check...")
    
    mask_indices = torch.zeros((8, 1), dtype=torch.bool, device=device)
    
    for epoch in range(101):
        batch = next(iter(loader))
        slots = batch["slots"].to(device)
        aux = batch["aux"].to(device)
        
        optimizer.zero_grad()
        outputs = model(slots, aux=aux, mask_indices=mask_indices)
        loss_dict = criterion(outputs["pred_slots"], slots, mask_indices)
        
        loss = loss_dict["loss_total"]
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")
            
    # Inference
    print("\nInference on Training Data:")
    with torch.no_grad():
        outputs = model(slots, aux=aux, mask_indices=mask_indices)
        pred = outputs["pred_slots"] # (B, T, 1, D)
        
        # Decode first sample
        t_vecs = slots[0, :, 0, :]
        p_vecs = pred[0, :, 0, :]
        
        def decode(vecs):
            dists = torch.cdist(vecs, embedding.to(device))
            ids = torch.argmin(dists, dim=1)
            return "".join([itos[i.item()] for i in ids])
            
        print(f"Target:    {decode(t_vecs)}")
        print(f"Predicted: {decode(p_vecs)}")

if __name__ == "__main__":
    main()
