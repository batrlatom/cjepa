
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class ModelConfig:
    """Hyperparameters for the C-JEPA model."""
    slot_dim: int = 128
    model_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 6
    mlp_dim: int = 2048
    dropout: float = 0.1
    aux_dim: int = 0  # Dimension of aux input (e.g. action+proprio). 0 if handled by separate encoder
    aux_embed_dim: int = 1024 # Project aux to this dim
    
    # Input data specs
    num_slots: int = 4 # Push-T: 4, CLEVRER: 7
    history_len: int = 3 # Push-T
    future_len: int = 1 # Push-T

@dataclass
class TrainConfig:
    """Hyperparameters for training."""
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.05
    epochs: int = 30
    warmup_epochs: int = 2
    clip_grad: float = 1.0
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Masking
    mask_ratio_min: int = 0
    mask_ratio_max: int = 2 # Max objects to mask (inclusive)
    
    # Logging
    log_freq: int = 10
    save_freq: int = 5
    run_name: str = "cjepa_pusht"
    output_dir: str = "runs"

@dataclass
class DataConfig:
    """Data paths and settings."""
    dataset_path: str = "data/pusht_slots.pt" # Placeholder
    # Provide dummy mode for reproduction without full dataset
    dummy_mode: bool = False 
    
    # Input shapes for dummy mode
    dummy_batch_size: int = 256
    input_slots_shape: Tuple[int, int, int] = (3+1, 4, 128) # (T_total, N, D)
    input_aux_shape: Tuple[int, int] = (3+1, 6) # (T_total, D_aux) e.g. action(2)+proprio(4)

@dataclass
class NLPConfig(ModelConfig):
    """Override for small NLP demo."""
    slot_dim: int = 64
    model_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    mlp_dim: int = 512
    dropout: float = 0.1
    aux_dim: int = 0
    # Input
    num_slots: int = 4 # Parallel chars
    history_len: int = 8
    future_len: int = 4

@dataclass
class ReasoningConfig(ModelConfig):
    """Config for Dyck-2 Parenthesis task."""
    slot_dim: int = 64
    model_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    mlp_dim: int = 512
    dropout: float = 0.0 # Deterministic task, turn off dropout
    aux_dim: int = 0
    # Input
    num_slots: int = 1 # Sequential
    history_len: int = 16 # Open sequence
    future_len: int = 16 # Close sequence (max)

@dataclass
class MazeConfig(ModelConfig):
    """Config for Fixed Maze Navigation task."""
    slot_dim: int = 64
    model_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    mlp_dim: int = 512
    dropout: float = 0.0 # Deterministic
    aux_dim: int = 64 # Goal embedding (same as slot_dim)
    aux_embed_dim: int = 256 # Project aux to model_dim
    # Input
    num_slots: int = 1 # Position
    history_len: int = 8 # Context
    future_len: int = 8 # Prediction path

@dataclass
class DynamicMazeConfig(ModelConfig):
    """Config for Dynamic Maze Navigation task (15x15)."""
    slot_dim: int = 64
    model_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    mlp_dim: int = 512
    dropout: float = 0.0
    aux_dim: int = 0 # Unused
    # Input is the full grid (flattened 15x15)
    num_slots: int = 225 
    history_len: int = 225 # We treat the whole grid as history context
    future_len: int = 0 # Prediction is masked token in history
    # Or, we can set history=0, future=225?
    # C-JEPA treats slots as [History... Future...]
    # If we provide 225 slots, and mask some.
    # The forward pass splits them into history/future based on indices.
    # But for MIM, we don't necessarily have a causal split.
    # We can just say history_len = 225. 
    # And mask_indices will determine what is predicted.

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    data: DataConfig = DataConfig()
    dataset_type: str = "pusht" # "pusht", "nlp", "reasoning", "maze"
