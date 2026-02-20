import math
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class CJEPAConfig:
    # Environment configs (Robomimic)
    action_dim: int = 7     # Control dimensions
    proprio_dim: int = 14   # Proprioception state dim
    vision_dim: int = 512   # CNN flat output dim
    
    # Model configs
    D: int = 256            # hidden dimension (transformer & latents)
    n_heads: int = 8
    n_layers: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1
    
    # Sequence Horizons
    obs_horizon: int = 2    # observations fed into context
    pred_horizon: int = 16  # future unobserved actions/states to predict

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden_dim = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x: (B, T, D)
        # mask: (T, T) for causal masking if needed, else None
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class CJEPA(nn.Module):
    """
    Continuous Joint Embedding Predictive Architecture Policy.
    Given an observation sequence of context latents over time, predict future states and decode them into actions.
    """
    def __init__(self, cfg: CJEPAConfig):
        super().__init__()
        self.cfg = cfg
        
        # Total context dimension fed to the model
        context_dim = cfg.vision_dim + cfg.proprio_dim
        
        # 1. Input Embedding (Context -> Model Dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.context_token_proj = nn.Linear(context_dim, cfg.D)
        
        # 2. Time Positional Encodings
        # We need embeddings for all tokens up to obs_horizon
        self.time_pos_embed = nn.Parameter(torch.randn(1, cfg.obs_horizon, cfg.D) * 0.02)
        
        # 3. Context Encoder Transformer
        self.context_encoder = nn.ModuleList([
            TransformerBlock(cfg.D, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.n_layers)
        ])
        
        # 4. Action Prediction Head (Decoder)
        # We predict a sequence of actions from a centralized summary latent
        self.action_decoder = nn.Sequential(
            nn.Linear(cfg.D, cfg.D),
            nn.GELU(),
            nn.Linear(cfg.D, cfg.pred_horizon * cfg.action_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def encode_context(self, vision_feats: torch.Tensor, proprio_feats: torch.Tensor) -> torch.Tensor:
        """
        vision_feats: (B, T_obs, 512)
        proprio_feats: (B, T_obs, 14)
        Returns flattened fused latents ready for transformer.
        """
        # Ensure they match time wise
        ctx = torch.cat([vision_feats, proprio_feats], dim=-1)
        return self.context_norm(ctx)

    def forward(self, context_latent: torch.Tensor) -> torch.Tensor:
        """
        Main policy network prediction flow.
        Input: context_latent (B, T_obs, context_dim)
        Output: actions_pred (B, T_pred, action_dim)
        """
        B, T_obs, _ = context_latent.shape
        
        # 1. Project to model dimension
        tokens = self.context_token_proj(context_latent) # (B, T_obs, D)
        
        # 2. Add time positional PE
        tokens = tokens + self.time_pos_embed[:, :T_obs, :]
        
        # 3. Pass through Context Encoder
        for block in self.context_encoder:
            tokens = block(tokens)
            
        # 4. Pooling: Mean over the observed time horizon to create a unified intent state
        intent_summary = tokens.mean(dim=1) # (B, D)
        
        # 5. Decode intent directly into future action trajectory
        # Note: In full JEPA, we would predict future latent states. 
        # Here we do direct implicit behavior cloning decoding from the unified state.
        actions_flat = self.action_decoder(intent_summary) # (B, pred_horizon * action_dim)
        actions = actions_flat.view(B, self.cfg.pred_horizon, self.cfg.action_dim)
        
        # We enforce tanh bound to match standard min-max action normalization [-1, 1]
        return torch.tanh(actions)
