"""Transformer + AdaLN backbone for flow matching (DiT-style).

Drop-in replacement for DiffusionConditionalUnet1d.
Same forward signature: forward(sample, timestep, global_cond) -> velocity.

Architecture follows Peebles & Xie (2023) "Scalable Diffusion Models with Transformers"
adapted for 1D action sequences. VITA (ICLR 2026) uses this architecture for their FM
baseline, achieving 100% on ALOHA CubeTransfer.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.cos(), args.sin()], dim=-1)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization — generates (scale, shift) from conditioning."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.linear = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.linear(F.silu(cond)).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.adaln = AdaLN(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.adaln(x, cond)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x


class FlowMatchingTransformer(nn.Module):
    """DiT-style Transformer for flow matching velocity prediction.

    Same forward(sample, timestep, global_cond) interface as DiffusionConditionalUnet1d.
    """

    def __init__(
        self,
        action_dim: int,
        global_cond_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_horizon: int = 64,
    ):
        super().__init__()
        self.d_model = d_model

        self.action_proj = nn.Linear(action_dim, d_model)
        self.cond_proj = nn.Linear(global_cond_dim, d_model)

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.pos_emb = nn.Embedding(max_horizon, d_model)

        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)])

        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_adaln = nn.Linear(d_model, 2 * d_model)
        self.output_proj = nn.Linear(d_model, action_dim)

        # AdaLN-zero init (DiT paper): start by predicting zero velocity
        nn.init.zeros_(self.final_adaln.weight)
        nn.init.zeros_(self.final_adaln.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, sample: Tensor, timestep: Tensor, global_cond: Tensor) -> Tensor:
        batch_size, seq_len, _ = sample.shape

        x = self.action_proj(sample)
        pos = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = x + pos.unsqueeze(0)

        t_emb = self.time_emb(timestep)
        c_emb = self.cond_proj(global_cond)
        cond = t_emb + c_emb

        # Prepend conditioning token
        x = torch.cat([cond.unsqueeze(1), x], dim=1)

        for block in self.blocks:
            x = block(x, cond)

        x = x[:, 1:, :]  # remove conditioning token

        scale, shift = self.final_adaln(F.silu(cond)).unsqueeze(1).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        return self.output_proj(x)
