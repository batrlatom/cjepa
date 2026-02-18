
import torch
from torch.utils.data import Dataset
import logging

class PushTDataset(Dataset):
    """
    Dataset for Push-T task.
    Supports either loading from disk or generating dummy data.
    """
    def __init__(self, config, split="train"):
        self.cfg = config
        self.split = split
        self.dummy_mode = config.dummy_mode
        self.data_path = config.dataset_path
        
        if self.dummy_mode:
            logging.info(f"Using Dummy Data for {split} split.")
            self.data = self._generate_dummy_data()
        else:
            try:
                self.data = torch.load(self.data_path)
            except FileNotFoundError:
                logging.warning(f"Dataset not found at {self.data_path}. Falling back to Dummy Data.")
                self.dummy_mode = True
                self.data = self._generate_dummy_data()
        
    def _generate_dummy_data(self):
        """
        Generates random dummy data for testing.
        """
        B = self.cfg.dummy_batch_size
        T = self.cfg.input_slots_shape[0]
        N = self.cfg.input_slots_shape[1]
        D = self.cfg.input_slots_shape[2]
        
        # Random slots
        slots = torch.randn(B, T, N, D)
        
        # Random aux (e.g. actions)
        aux_dim = self.cfg.input_aux_shape[1]
        aux = torch.randn(B, T, aux_dim)
        
        return {
            "slots": slots,
            "aux": aux
        }

    def __len__(self):
        if self.dummy_mode:
            return self.cfg.dummy_batch_size
        return len(self.data["slots"])

    def __getitem__(self, idx):
        return {
            "slots": self.data["slots"][idx],
            "aux": self.data["aux"][idx]
        }
