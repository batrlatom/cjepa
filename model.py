from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CJEPAConfig:
    # Paper notation: N objects, latent slot dim d, model dim D.
    N: int = 81
    d: int = 10  # Vocab size for discrete logits (digits 0-9)
    D: int = 256
    n_heads: int = 8
    n_layers: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # History length T_h and prediction horizon T_p.
    T_h: int = 1
    T_p: int = 0
    aux_dim: int = 0

    @property
    def T_total(self) -> int:
        return self.T_h + self.T_p


class TransformerBlock(nn.Module):
    def __init__(self, D: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(embed_dim=D, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, int(D * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(D * mlp_ratio), D),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class CJEPA(nn.Module):
    """
    Sudoku-adapted C-JEPA latent predictor.
    Uses discrete embedding and spatial encoding.
    """

    def __init__(self, cfg: CJEPAConfig):
        super().__init__()
        self.cfg = cfg

        self.slot_embed = nn.Embedding(cfg.d, cfg.D)
        
        # Spatial positional encoding for elements (e.g. 81 cells)
        self.pos_pe = nn.Embedding(cfg.N, cfg.D)

        # Mask embedding for predicting hidden states
        self.e_mask = nn.Parameter(torch.zeros(1, 1, 1, cfg.D))

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.D, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.D)
        
        # Classification head
        self.head = nn.Linear(cfg.D, cfg.d)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.e_mask, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_mask_map(self, B: int, T: int, N: int, M: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
        """
        Build boolean indicator mask map.
        M format:
          - None: no masking
          - (B, N): mask indices over objects
          - (B, T, N): explicit mask
        """
        if M is None:
            mask = torch.zeros((B, T, N), dtype=torch.bool, device=device)
        elif M.dim() == 2:
            mask = M.unsqueeze(1).expand(B, T, N).bool()
        elif M.dim() == 3:
            mask = M.bool()
        else:
            raise ValueError(f"Expected M to have dim 2 or 3, got shape {tuple(M.shape)}")
        return mask

    def forward(
        self,
        z: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        M: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            z: discrete inputs, shape (B, T, N)
            u: not used in Sudoku
            M: mask indices over objects

        Returns:
            dict with z_hat, z_bar, and mask_map.
        """
        B, T, N = z.shape[:3]
        if T != self.cfg.T_total:
            raise ValueError(f"Expected T={self.cfg.T_total}, got T={T}")

        pos_idx = torch.arange(N, device=z.device).view(1, 1, N)
        pos_embed = self.pos_pe(pos_idx)  # (1, 1, N, D)

        z_tok = self.slot_embed(z) + pos_embed  # tokenized and spatially encoded

        mask_map = self._build_mask_map(B, T, N, M, z.device)
        
        z_tilde = self.e_mask + pos_embed

        z_bar = torch.where(mask_map.unsqueeze(-1), z_tilde, z_tok)

        tokens = z_bar

        B2, T2, N_plus, D = tokens.shape
        x = tokens.view(B2, T2 * N_plus, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.view(B2, T2, N_plus, D)

        z_hat = self.head(x[:, :, :N, :])

        return {
            "z_hat": z_hat,
            "z_bar": z_bar,
            "mask_map": mask_map,
        }
