"""Two-tower fusion encoder for continuous + token inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class FusionConfig:
    continuous_dim: int
    token_vocab_size: int
    d_model: int = 128
    nheads: int = 4
    depth: int = 2
    dropout: float = 0.1
    token_embed_dim: int = 64


class FusionBackbone(nn.Module):
    """Fuse continuous OHLCV features with discrete price-action tokens."""

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        self.continuous_proj = nn.Linear(config.continuous_dim, config.d_model)
        self.token_embedding = nn.Embedding(config.token_vocab_size, config.token_embed_dim)
        self.token_proj = nn.Linear(config.token_embed_dim, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nheads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, continuous: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        cont_emb = self.continuous_proj(continuous)
        token_emb = self.token_embedding(tokens)
        token_emb = self.token_proj(token_emb)
        fused = cont_emb + token_emb
        encoded = self.encoder(fused)
        return self.layer_norm(encoded)
