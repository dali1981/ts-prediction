"""Reference PatchTST implementation adapted from the public repo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(slots=True)
class PatchTSTReferenceConfig:
    input_dim: int
    patch_length: int = 16
    stride: int = 8
    d_model: int = 256
    nheads: int = 8
    depth: int = 6
    dropout: float = 0.2
    feedforward_dim: Optional[int] = None


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(x.dtype)


class PatchTSTReference(nn.Module):
    """Closer reproduction of the official PatchTST backbone."""

    def __init__(self, config: PatchTSTReferenceConfig) -> None:
        super().__init__()
        self.config = config
        ff_dim = config.feedforward_dim or config.d_model * 4

        self.instance_norm = nn.InstanceNorm1d(config.input_dim)
        self.patch_embed = nn.Conv1d(
            in_channels=config.input_dim,
            out_channels=config.d_model,
            kernel_size=config.patch_length,
            stride=config.stride,
            padding=0,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.positional = PositionalEncoding(config.d_model)

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
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        x = x.transpose(1, 2)  # (batch, features, time)
        x = self.instance_norm(x)
        patches = self.patch_embed(x)
        patches = patches.transpose(1, 2)  # (batch, patches, d_model)
        patches = self.dropout(patches)
        encoded = self.encoder(self.positional(patches))
        return self.layer_norm(encoded)


__all__ = ["PatchTSTReference", "PatchTSTReferenceConfig"]
