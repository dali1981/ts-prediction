"""Temporal Fusion Transformer style backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(slots=True)
class TemporalFusionTransformerConfig:
    input_dim: int
    d_model: int = 256
    nheads: int = 8
    depth: int = 4
    dropout: float = 0.1
    feedforward_dim: Optional[int] = None


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(x.dtype)


class TemporalFusionTransformerBackbone(nn.Module):
    """Simplified TFT encoder/decoder working with sliding-window data."""

    expects_future = True

    def __init__(self, config: TemporalFusionTransformerConfig) -> None:
        super().__init__()
        self.config = config
        ff_dim = config.feedforward_dim or config.d_model * 4

        self.past_proj = nn.Linear(config.input_dim, config.d_model)
        self.future_proj = nn.Linear(config.input_dim, config.d_model)
        self.past_pos = _PositionalEncoding(config.d_model)
        self.future_pos = _PositionalEncoding(config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nheads,
            dim_feedforward=ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nheads,
            dim_feedforward=ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.depth)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        # past: (batch, lookback, features)
        # future: (batch, horizon, features)
        past_emb = self.past_proj(past)
        future_emb = self.future_proj(future)

        past_encoded = self.encoder(self.past_pos(past_emb))
        decoded = self.decoder(
            self.future_pos(future_emb),
            past_encoded,
        )
        return self.layer_norm(decoded)


__all__ = [
    "TemporalFusionTransformerBackbone",
    "TemporalFusionTransformerConfig",
]
