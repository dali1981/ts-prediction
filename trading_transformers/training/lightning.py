"""PyTorch Lightning modules for forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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
        self._epoch_cache: Dict[str, Dict[str, list[torch.Tensor]]] = {
            "train": {"preds": [], "target": []},
            "val": {"preds": [], "target": []},
            "test": {"preds": [], "target": []},
        }

    def _encode(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, dict):
            continuous = inputs["continuous"]
            tokens = inputs.get("tokens")
            future = inputs.get("future")
            if hasattr(self.backbone, "expects_future") and getattr(self.backbone, "expects_future"):
                if future is None:
                    raise ValueError("Backbone expects future features but none provided")
                return self.backbone(continuous, future)
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
        self._store_batch("train", preds, target)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mae", mae, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        features, target = batch
        preds = self(features)
        loss = F.mse_loss(preds, target)
        mae = F.l1_loss(preds, target)
        self._store_batch("val", preds, target)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_mae": mae}

    def test_step(self, batch, batch_idx: int):
        features, target = batch
        preds = self(features)
        loss = F.mse_loss(preds, target)
        mae = F.l1_loss(preds, target)
        self._store_batch("test", preds, target)
        self.log("test_loss", loss)
        self.log("test_mae", mae)
        return {"test_loss": loss, "test_mae": mae}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_params.lr,
            weight_decay=self.optimizer_params.weight_decay,
        )

    # ------------------------------------------------------------------ epoch hooks
    def on_train_epoch_start(self) -> None:
        self._clear_buffers("train")

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_start(self) -> None:
        self._clear_buffers("val")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_test_epoch_start(self) -> None:
        self._clear_buffers("test")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")

    # ----------------------------------------------------------------- utilities
    def _store_batch(self, stage: str, preds: torch.Tensor, target: torch.Tensor) -> None:
        self._epoch_cache[stage]["preds"].append(preds.detach().cpu())
        self._epoch_cache[stage]["target"].append(target.detach().cpu())

    def _clear_buffers(self, stage: str) -> None:
        self._epoch_cache[stage]["preds"].clear()
        self._epoch_cache[stage]["target"].clear()

    def _log_epoch_metrics(self, stage: str) -> None:
        preds_list = self._epoch_cache[stage]["preds"]
        target_list = self._epoch_cache[stage]["target"]
        if not preds_list:
            return
        preds = torch.cat(preds_list, dim=0)
        target = torch.cat(target_list, dim=0)
        metrics = self._additional_metrics(preds, target)
        log_kwargs = dict(on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        for name, value in metrics.items():
            self.log(f"{stage}_{name}", value, **log_kwargs)

    def _additional_metrics(self, preds: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        preds_flat = preds.reshape(-1).float()
        target_flat = target.reshape(-1).float()

        eps = torch.finfo(preds_flat.dtype).eps
        diff = preds_flat - target_flat

        rmse = torch.sqrt(torch.mean(diff ** 2))
        mape = torch.mean(torch.abs(diff / (target_flat.abs() + eps)))
        bias = diff.mean()
        std_ratio = preds_flat.std(unbiased=False) / (target_flat.std(unbiased=False) + eps)

        sign_match = torch.sign(preds_flat) == torch.sign(target_flat)
        directional = sign_match.float().mean()

        target_mask = target_flat.abs() > eps
        if target_mask.any():
            hit_rate = sign_match[target_mask].float().mean()
        else:
            hit_rate = torch.tensor(0.0, device=preds.device, dtype=preds.dtype)

        preds_centered = preds_flat - preds_flat.mean()
        target_centered = target_flat - target_flat.mean()
        numerator = (preds_centered * target_centered).sum()
        denominator = torch.sqrt(preds_centered.pow(2).sum() * target_centered.pow(2).sum() + eps)
        corr = numerator / (denominator + eps)

        pred_up = preds_flat > 0
        pred_down = preds_flat < 0
        actual_up = target_flat > 0
        actual_down = target_flat < 0
        long_hits = (pred_up & actual_up).float().sum()
        short_hits = (pred_down & actual_down).float().sum()
        long_total = pred_up.float().sum()
        short_total = pred_down.float().sum()
        long_precision = long_hits / (long_total + eps)
        short_precision = short_hits / (short_total + eps)

        metrics = {
            "rmse": rmse,
            "mape": mape,
            "bias": bias,
            "std_ratio": std_ratio,
            "directional_acc": directional,
            "hit_rate": hit_rate,
            "corr": corr,
            "long_precision": long_precision,
            "short_precision": short_precision,
        }

        for key, value in metrics.items():
            metrics[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        return metrics
