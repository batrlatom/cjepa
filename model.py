
import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, Tuple

from cjepa.config import ModelConfig
from cjepa.blocks import Block, PositionalEncoding

logger = logging.getLogger(__name__)

class CJEPA(nn.Module):
    """
    Causal-JEPA Model.
    Architecture:
        - Slot Embedder: Linear projection of input slots (d_slot -> d_model)
        - Aux Embedder: Embeds auxiliary inputs (actions/proprio) -> d_model
        - Mask Token: Learnable token for masked slots.
        - Predictor: Transformer Encoder processing flat sequence of (slots + aux).
        - Head: Project back to d_slot.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cfg = config
        
        # 1. Embeddings
        # S -> S_embed
        self.slot_proj = nn.Linear(config.slot_dim, config.model_dim)
        
        # Aux -> Aux_embed
        if config.aux_dim > 0:
            # Simple linear projection for aux inputs
            self.aux_proj = nn.Linear(config.aux_dim, config.model_dim)
        else:
            self.aux_proj = None

        # Positional Encodings
        # We need temporal position encoding. 
        # T_total = history_len + future_len
        self.max_len = config.history_len + config.future_len
        self.temp_pe = PositionalEncoding(self.max_len, config.model_dim)
        
        # Mask Token & Anchor Projection
        # Paper says: tilde_z = phi(z_t0) + e_tau
        # phi: Linear(d_model -> d_model)
        self.anchor_proj = nn.Linear(config.model_dim, config.model_dim)
        
        # e_tau is learnable embedding per timestep. 
        # We can reuse temp_pe or have separate mask PEs.
        # Paper says "learnable embedding combined with temporal positional encoding".
        # Let's use a specific learnable mask vector *per relative timestep*.
        self.mask_tokens = nn.Parameter(torch.zeros(self.max_len, config.model_dim))
        nn.init.normal_(self.mask_tokens, std=0.02)
        
        # 2. Predictor
        self.blocks = nn.ModuleList([
            Block(
                dim=config.model_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_dim / config.model_dim,
                drop=config.dropout,
                attn_drop=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.model_dim)
        
        # 3. Prediction Head
        self.head = nn.Linear(config.model_dim, config.slot_dim)
        
        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, 
                slots: torch.Tensor, 
                aux: Optional[torch.Tensor] = None, 
                mask_indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            slots: (B, T, N, D_slot)
            aux: (B, T, D_aux) or None
            mask_indices: (B, N) boolean tensor. True means masked.
        
        Returns:
            pred_slots: (B, T, N, D_slot)
        """
        B, T, N, D = slots.shape
        
        # 1. Tokenize / Embed
        # (B, T, N, D_model)
        z_slots = self.slot_proj(slots)
        
        # Add Temporal PE to slots
        # shape (1, T, 1) -> broadcast to (B, T, N, D) implies each slot at time t gets same PE[t]
        t_indices = torch.arange(T, device=slots.device).unsqueeze(0) # (1, T)
        pe = self.temp_pe(t_indices) # (1, T, D_model)
        z_slots = z_slots + pe.unsqueeze(2)

        # 2. Apply Masking
        # IMPORTANT: We verify if T matches config
        assert T == self.max_len, f"Input sequence length {T} must match config {self.max_len}"
        
        if mask_indices is not None:
            # mask_indices: (B, N) -> selected objects are masked for all t > t0 ?
            # Paper: "preserving only the earliest time step t0 as an identity anchor"
            # So for masked objects k in M:
            #   z_{k, t0} is kept (anchor)
            #   z_{k, t} is REPLACED by phi(z_{k, t0}) + e_t for t > t0 (history) OR all t (future)?
            #   Usually history masking means t in history window > t0.
            #   Future is ALWAYS masked for prediction.
            
            # Let's implement robust masking:
            # Identify Anchor: t=0
            # Identify Filter:
            #   For Unmasked objects: Keep all z_{i, t}
            #   For Masked objects: Keep z_{i, 0} (anchor). Replace z_{i, t} (t>0) with mask token.
            
            # Construct Mask Tokens
            # phi(z_{i, 0})
            z_anchors = z_slots[:, 0, :, :] # (B, N, D_model)
            anchor_proj = self.anchor_proj(z_anchors) # (B, N, D_model)
            
            # e_tau
            # (1, T, 1, D)
            mask_embeds = self.mask_tokens.unsqueeze(0).unsqueeze(2) # (1, T, 1, D)
            
            # Combine: phi(anchor) + e_tau -> (B, T, N, D)
            # anchor_proj is (B, N, D), broadcast to T
            generated_mask_tokens = anchor_proj.unsqueeze(1) + mask_embeds
            
            # Apply replacement
            # Mask is (B, N). Expand to (B, T, N, D)
            # We want to mask where mask_indices is True AND time > 0
            
            # Future masking: implied?
            # The paper says: "Future entity tokens are always masked for prediction"
            # It implies ALL objects are masked in future? Or just target objects?
            # Usually world models predict ALL future states.
            # "At inference time, f is used solely for forward prediction... masking only future tokens."
            # So for training L_future, we mask ALL objects at t > history_len.
            
            # Let's refine the mask logic:
            # Check mask shape
            if mask_indices.dim() == 3: # (B, T, N)
                final_mask = mask_indices
            else:
                # (B, N) -> (B, 1, N)
                m_obj = mask_indices.unsqueeze(1) 
                
                # Time indices
                time_indices = torch.arange(T, device=slots.device).reshape(1, T, 1) # (1,T,1)
                is_future = time_indices >= self.cfg.history_len
                is_history_mask = (time_indices > 0) & (time_indices < self.cfg.history_len) & m_obj
                
                final_mask = is_future | is_history_mask # (B, T, N) boolean
            
            # Expand to D
            final_mask_expanded = final_mask.unsqueeze(-1)
            
            # Replace
            z_slots = torch.where(final_mask_expanded, generated_mask_tokens, z_slots)
            
            # Debug/Logging could happen here
            
        # 3. Aux handling
        # Aux is (B, T, D_conv) or similar.
        # We embed it and concat as extra token(s)?
        # "U_t ... is represented as a set of entity tokens Z_t = {S_t, U_t}"
        # So at each timestep t, we have N slot tokens + 1 aux token.
        
        tokens = z_slots # (B, T, N, D)
        
        if self.aux_proj is not None and aux is not None:
             z_aux = self.aux_proj(aux) # (B, T, D_model)
             z_aux = z_aux + pe # Add PE to aux too
             
             # Concat: (B, T, N+1, D)
             # We stack in dim 2
             z_aux = z_aux.unsqueeze(2) # (B, T, 1, D)
             tokens = torch.cat([tokens, z_aux], dim=2)
             
        # Flatten for Transformer: (B, T*(N+1), D)
        B, T, N_plus, D = tokens.shape
        x = tokens.view(B, T * N_plus, D)
        
        # 4. Transformer Forward
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        # 5. Extract output slots
        # Reshape back: (B, T, N+1, D)
        x = x.view(B, T, N_plus, D)
        
        # Slice only the object slots (exclude aux)
        # Assuming aux was appended at end
        x_slots = x[:, :, :N, :] # (B, T, N, D)
        
        # Project back to slot dim
        pred_slots = self.head(x_slots)
        
        return {
            "pred_slots": pred_slots, # (B, T, N, D_slot)
            "mask_indices": mask_indices # Pass through for loss
        }
