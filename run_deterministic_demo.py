
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
import logging

# Re-use our modular code
from cjepa.config import NLPConfig
from cjepa.model import CJEPA
from cjepa.loss import CJEPALoss

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class DeterministicDataset(Dataset):
    """
    Infinite dataset of a repeated deterministic sequence.
    """
    def __init__(self, seq_len: int, slot_dim: int):
        self.text = "the quick brown fox jumps over the lazy dog. " * 50
        self.seq_len = seq_len
        self.chars = sorted(list(set(self.text)))
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for i,ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # Fixed random embeddings
        # Use a larger scale to avoid collapse to zero?
        self.embedding = torch.randn(self.vocab_size, slot_dim) * 5.0
        
        self.data_ids = [self.stoi[c] for c in self.text]

    def __len__(self):
        return 1000 # items per epoch

    def __getitem__(self, idx):
        # Return a random window
        # For meaningful learning, let's just cycle through
        start = (idx * 5) % (len(self.data_ids) - self.seq_len)
        chunk_ids = self.data_ids[start : start + self.seq_len]
        
        # (T, 1, D)
        slots = self.embedding[chunk_ids].unsqueeze(1)
        aux = torch.zeros(self.seq_len, 1) # Dummy
        
        return {"slots": slots, "aux": aux}
        
    def decode(self, vecs):
        # vecs: (T, D)
        dists = torch.cdist(vecs, self.embedding.to(vecs.device))
        ids = torch.argmin(dists, dim=1)
        return "".join([self.itos[i.item()] for i in ids])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Deterministic Demo on {device}...")
    
    # Tiny Config
    cfg = NLPConfig()
    cfg.history_len = 8
    cfg.future_len = 8
    cfg.num_slots = 1
    cfg.model_dim = 256
    cfg.num_layers = 4
    cfg.slot_dim = 32
    
    # Dataset
    T = cfg.history_len + cfg.future_len
    dataset = DeterministicDataset(seq_len=T, slot_dim=cfg.slot_dim)
    loader = DataLoader(dataset, batch_size=32)
    
    # Model
    model = CJEPA(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = CJEPALoss(history_len=cfg.history_len)
    
    # Train Loop
    epochs = 500
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            slots = batch["slots"].to(device)
            aux = batch["aux"].to(device)
            
            current_bs = slots.shape[0]
            mask_indices = torch.zeros((current_bs, 1), dtype=torch.bool, device=device)
            
            optimizer.zero_grad()
            outputs = model(slots, aux=aux, mask_indices=mask_indices)
            loss_dict = criterion(outputs["pred_slots"], slots, mask_indices)
            loss = loss_dict["loss_total"]
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {avg_loss:.4f}")
            
    print("Training Complete. Visualizing Predictions...")
    
    # Visualize
    model.eval()
    with torch.no_grad():
        # Retrieve a clear sample (start of sentence)
        sample_ids = [dataset.stoi[c] for c in "the quick brown "]
        slots = dataset.embedding[sample_ids].unsqueeze(1).unsqueeze(0).to(device) # (1, 16, 1, D)
        aux = torch.zeros(1, 16, 1).to(device)
        mask = torch.zeros((1, 1), dtype=torch.bool, device=device)
        
        outputs = model(slots, aux=aux, mask_indices=mask)
        pred = outputs["pred_slots"]
        
        # Decode
        # History: first 8
        # Future: last 8
        
        input_seq = dataset.decode(slots[0, :, 0, :])
        pred_seq = dataset.decode(pred[0, :, 0, :])
        
        print("\nInput Sequence:     last 'quick' -> predict 'brown '")
        print(f"Full Input:         '{input_seq}'")
        print(f"Full Predicted:     '{pred_seq}'")
        
        # Check alignment
        print("\nAlignment:")
        print(f"Context (0-8):      '{input_seq[:8]}'")
        print(f"Future (Target):    '{input_seq[8:]}'")
        print(f"Future (Pred):      '{pred_seq[8:]}'")

if __name__ == "__main__":
    main()
