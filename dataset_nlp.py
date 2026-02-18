
import torch
from torch.utils.data import Dataset
import os
import requests
import logging

class NLPDataset(Dataset):
    """
    Adapts text data to C-JEPA 'Slot' interface.
    
    Concept:
    - Text is a 1D sequence of characters.
    - We map each char to a dense vector (random fixed embedding).
    - We reshape the sequence into (T_total, N_slots) chunks.
    - Each 'slot' is a character position in the chunk.
    - 'Time' flows down the sequence. 'Slots' are parallel characters.
    """
    def __init__(self, config, split="train", download=True):
        self.cfg = config
        self.split = split
        self.block_size = (config.history_len + config.future_len) * config.num_slots
        self.data_path = "data/tiny_shakespeare.txt"
        
        if download and not os.path.exists(self.data_path):
            self._download_data()
            
        with open(self.data_path, 'r') as f:
            text = f.read()
            
        # Simple Char Tokenizer
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        
        data_ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        
        # Split (90/10)
        n = int(0.9 * len(data_ids))
        self.data = data_ids[:n] if split == "train" else data_ids[n:]
        
        # Fixed Random Embedding (Simulating "pre-trained slots")
        # seed specific to ensure consistency
        g = torch.Generator()
        g.manual_seed(42)
        self.embedding = torch.randn(self.vocab_size, config.slot_dim, generator=g)
        
        # Stride: defaults to block_size for non-overlapping (faster)
        self.stride = self.block_size
        
        # Log
        logging.info(f"NLP Dataset ({split}) loaded. Length: {len(self.data)} chars. Vocab: {self.vocab_size}. Stride: {self.stride}")

    def _download_data(self):
        os.makedirs("data", exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        logging.info(f"Downloading {url}...")
        res = requests.get(url)
        with open(self.data_path, 'w') as f:
            f.write(res.text)
            
    def __len__(self):
        # Number of chunks
        if len(self.data) < self.block_size:
            return 0
        return (len(self.data) - self.block_size) // self.stride + 1

    def __getitem__(self, idx):
        # Grab chunk
        start = idx * self.stride
        chunk = self.data[start : start + self.block_size]
        
        # Reshape to (T, N)
        # T = history + future
        # N = num_slots
        T_total = self.cfg.history_len + self.cfg.future_len
        N = self.cfg.num_slots
        
        # Check if we have enough data (boundary case handled by __len__)
        
        # Reshape (Chunk) -> (T, N)
        # We fill row by row? or col by col?
        # Row by row is more "temporal":
        # T=0: chars[0..N]
        # T=1: chars[N..2N]
        # This implies "Slots" correspond to "Position 1, Position 2..." in a window.
        
        chunk = chunk.view(T_total, N)
        
        # Embed
        # (T, N) -> (T, N, D)
        slots = self.embedding[chunk]
        
        # Aux: Dummy zeros (No actions in Shakespeare)
        aux_dim = self.cfg.aux_dim # Expected from config?
        # If model expects 0 aux, this is fine.
        # But if model config has aux_dim > 0 by default, we need to supply it.
        # Let's assume Config.aux_dim defaults to 0 for NLP or we set it to 0.
        
        aux = torch.zeros(T_total, max(1, self.cfg.aux_dim)) # Dummy
        
        return {
            "slots": slots,
            "aux": aux
        }
