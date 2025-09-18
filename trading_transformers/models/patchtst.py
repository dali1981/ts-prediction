"""PatchTST-inspired backbone in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(slots=True)
class PatchTSTConfig:
    input_dim: int
    patch_length: int = 16
    stride: int = 8
    d_model: int = 128
    nheads: int = 8
    depth: int = 4
    dropout: float = 0.1


class PatchTSTBackbone(nn.Module):
    """Minimal PatchTST encoder for experimentation."""

    def __init__(self, config: PatchTSTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv1d(
            in_channels=config.input_dim,
            out_channels=config.d_model,
            kernel_size=config.patch_length,
            stride=config.stride,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nheads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, timesteps, features)
        x = x.transpose(1, 2)
        patches = self.patch_embed(x)
        patches = patches.transpose(1, 2)
        encoded = self.encoder(patches)
        return self.layer_norm(encoded)
