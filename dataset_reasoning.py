
import torch
from torch.utils.data import Dataset
import random
import logging

class ReasoningDataset(Dataset):
    """
    Generates balanced parenthesis sequences (Dyck-2 language).
    Vocabulary: ( ) [ ] { }
    Task: Given an open sequence, predict the valid closing sequence.
    Example: 
      Input:  ( [ {
      Target: } ] )
      
    This tests the model's ability to maintain a stack.
    """
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        self.length = 10000 if split == "train" else 1000
        
        # Vocab: 
        # 0: (
        # 1: )
        # 2: [
        # 3: ]
        # 4: {
        # 5: }
        # 6: PAD
        self.pairs = {0: 1, 2: 3, 4: 5}
        self.opens = [0, 2, 4]
        self.closes = [1, 3, 5]
        
        self.vocab_size = 7
        self.chars = ["(", ")", "[", "]", "{", "}", "_"]
        self.stoi = {c:i for i,c in enumerate(self.chars)}
        self.itos = {i:c for i,c in enumerate(self.chars)}
        
        # Fixed Embeddings
        g = torch.Generator()
        g.manual_seed(1337)
        self.embedding = torch.randn(self.vocab_size, config.slot_dim, generator=g)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Generate random open sequence
        # Length between 1 and history_len
        seq_len = random.randint(1, self.config.history_len)
        
        stack = []
        open_seq = []
        
        for _ in range(seq_len):
            op = random.choice(self.opens)
            open_seq.append(op)
            stack.append(op)
            
        # Target closing sequence is just popping the stack
        close_seq = []
        while stack:
            op = stack.pop()
            close_seq.append(self.pairs[op])
            
        # Pad both to fixed lengths
        # History: Pad at start (left pad) or end? 
        # Let's pad at END for simplicity, but masked transformer handles position.
        
        # Input: Open seq + Padding
        slots_in = [self.embedding[t] for t in open_seq]
        pad_vec = self.embedding[6]
        
        # Pad input to history_len
        while len(slots_in) < self.config.history_len:
            slots_in.append(pad_vec)
            
        # Target: Close seq + Padding
        slots_out = [self.embedding[t] for t in close_seq]
        while len(slots_out) < self.config.future_len:
            slots_out.append(pad_vec)
            
        # Construct Tensors
        # (T, N, D)
        # Combine history and future
        
        full_seq = slots_in + slots_out
        # Truncate if exceeds total (shouldn't by logic above)
        
        full_tensor = torch.stack(full_seq).unsqueeze(1) # (T, 1, D)
        
        return {
            "slots": full_tensor,
            "aux": torch.zeros(len(full_seq), 1)
        }

    def decode(self, vecs):
        # vecs: (T, D)
        dists = torch.cdist(vecs, self.embedding.to(vecs.device))
        ids = torch.argmin(dists, dim=1)
        return "".join([self.itos[i.item()] for i in ids])
