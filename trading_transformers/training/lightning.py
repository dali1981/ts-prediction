"""PyTorch Lightning modules for forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("pytorch_lightning must be installed to use lightning modules") from exc


@dataclass(slots=True)
class OptimizerParams:
    lr: float
    weight_decay: float


class ForecastingModule(pl.LightningModule):
    """Generic forecasting head on top of a backbone encoder."""

    def __init__(self, backbone: nn.Module, horizon: int, optimizer_params: OptimizerParams) -> None:
        super().__init__()
        self.backbone = backbone
        self.horizon = horizon
        self.optimizer_params = optimizer_params
        if not hasattr(backbone, "config") or not hasattr(backbone.config, "d_model"):
            raise AttributeError("Backbone must expose config.d_model for head initialisation")
        self.head = nn.Linear(backbone.config.d_model, horizon)

    def _encode(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, dict):
            continuous = inputs["continuous"]
            tokens = inputs.get("tokens")
            if tokens is not None:
                return self.backbone(continuous, tokens)
            return self.backbone(continuous)
        if isinstance(inputs, (tuple, list)):
            return self.backbone(*inputs)
        return self.backbone(inputs)

    def forward(self, inputs: Any) -> torch.Tensor:
        encoded = self._encode(inputs)
        pooled = encoded[:, -1, :]
        return self.head(pooled)

    def training_step(self, batch, batch_idx: int):
        features, target = batch
        preds = self(features)
        loss = F.mse_loss(preds, target)
        mae = F.l1_loss(preds, target)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mae", mae, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        features, target = batch
        preds = self(features)
        loss = F.mse_loss(preds, target)
        mae = F.l1_loss(preds, target)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx: int):
        features, target = batch
        preds = self(features)
        loss = F.mse_loss(preds, target)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_params.lr,
            weight_decay=self.optimizer_params.weight_decay,
        )
