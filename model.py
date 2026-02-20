from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CJEPAConfig:
    # Paper notation: N objects, latent slot dim d, model dim D.
    N: int = 4
    d: int = 128
    D: int = 256
    n_heads: int = 8
    n_layers: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # History length T_h and prediction horizon T_p.
    T_h: int = 3
    T_p: int = 1

    # Optional auxiliary variables U_t.
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
    Minimal C-JEPA latent predictor.

    Eq. (4): Z_hat_T = f(Z_bar_T)
    Eq. (3): z_tilde_tau^i = phi(z_t0^i) + e_tau
    """

    def __init__(self, cfg: CJEPAConfig):
        super().__init__()
        self.cfg = cfg

        self.slot_proj = nn.Linear(cfg.d, cfg.D)
        self.aux_proj = nn.Linear(cfg.aux_dim, cfg.D) if cfg.aux_dim > 0 else None

        # phi in Eq. (3), applied to identity anchor z_{t0}^i.
        self.phi = nn.Linear(cfg.D, cfg.D)

        # e_tau in Eq. (3), plus temporal position encoding.
        self.e_tau = nn.Parameter(torch.zeros(cfg.T_total, cfg.D))
        self.tau_pe = nn.Embedding(cfg.T_total, cfg.D)

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.D, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.D)
        self.head = nn.Linear(cfg.D, cfg.d)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.e_tau, std=0.02)
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
        Build mask indicator 1[z_bar_tau^i != z_tau^i] used in Eq. (5).

        M format:
          - None: no history object masking (future-only masking at inference).
          - (B, N): object mask indices for history window.
          - (B, T, N): explicit mask map.
        """
        tau = torch.arange(T, device=device).view(1, T, 1)
        future_mask = (tau >= self.cfg.T_h).expand(B, T, N)

        if M is None:
            history_mask = torch.zeros((B, T, N), dtype=torch.bool, device=device)
        elif M.dim() == 2:
            history_mask = (tau > 0) & (tau < self.cfg.T_h) & M.unsqueeze(1)
        elif M.dim() == 3:
            history_mask = M.bool()
        else:
            raise ValueError(f"Expected M to have dim 2 or 3, got shape {tuple(M.shape)}")

        return history_mask | future_mask

    def forward(
        self,
        z: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        M: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            z: object latents Z, shape (B, T, N, d)
            u: optional auxiliaries U, shape (B, T, d_u)
            M: mask indices over objects/tokens (see _build_mask_map)

        Returns:
            dict with z_hat, z_bar, and mask_map.
        """
        B, T, N, _ = z.shape
        if T != self.cfg.T_total:
            raise ValueError(f"Expected T={self.cfg.T_total}, got T={T}")

        tau_idx = torch.arange(T, device=z.device).view(1, T)
        tau_embed = self.tau_pe(tau_idx).unsqueeze(2)  # (1, T, 1, D)

        z_tok = self.slot_proj(z) + tau_embed  # tokenized Z

        # Eq. (3): masked token z_tilde_tau^i = phi(z_t0^i) + e_tau.
        mask_map = self._build_mask_map(B, T, N, M, z.device)
        z_t0 = z_tok[:, 0]  # earliest time step t0
        z_tilde = self.phi(z_t0).unsqueeze(1) + self.e_tau.view(1, T, 1, self.cfg.D) + tau_embed

        z_bar = torch.where(mask_map.unsqueeze(-1), z_tilde, z_tok)

        tokens = z_bar
        if self.aux_proj is not None and u is not None:
            u_tok = self.aux_proj(u) + self.tau_pe(tau_idx)
            tokens = torch.cat([tokens, u_tok.unsqueeze(2)], dim=2)

        # Eq. (4): Z_hat_T = f(Z_bar_T)
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


def sample_object_mask(B: int, N: int, m_min: int, m_max: int, device: torch.device) -> torch.Tensor:
    """Sample M ~ Uniform({1,...,N}) style object masks used in C-JEPA training."""
    k = torch.randint(low=m_min, high=m_max + 1, size=(B,), device=device)
    M = torch.zeros((B, N), dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(N, device=device)[: k[b]]
        M[b, idx] = True
    return M
