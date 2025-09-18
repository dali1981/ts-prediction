"""Experiment runner orchestrating data, model, and trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch.nn as nn

try:
    import pytorch_lightning as pl
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("pytorch_lightning must be installed to use ExperimentRunner") from exc

from ..data import DataCatalog
from ..models import FusionBackbone, FusionConfig, PatchTSTBackbone, PatchTSTConfig
from ..tokenizers import BrooksTokenVocabulary
from ..evaluation.diagnostics import fusion_token_report
from .config import ExperimentConfig
from .datamodule import DataModuleBuilder
from .lightning import ForecastingModule, OptimizerParams


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        catalog: DataCatalog,
        data_frame: Optional[pd.DataFrame] = None,
    ) -> None:
        self.config = config
        self.catalog = catalog
        self._frame = data_frame
        self._report: dict[str, object] = {}
        self._datamodule = None

    def load_frame(self) -> pd.DataFrame:
        if self._frame is not None:
            return self._frame
        self._frame = self.catalog.load(self.config.data.source)
        return self._frame

    def _vocab_size(self) -> Optional[int]:
        vocab_path = self.config.data.vocab_path
        if not vocab_path:
            return None
        vocab = BrooksTokenVocabulary.from_json(Path(vocab_path))
        return len(vocab.tokens)

    def build_backbone(self) -> nn.Module:
        model_cfg = self.config.model
        model_type = model_cfg.get("type", "patchtst")
        if model_type == "patchtst":
            patch_config = PatchTSTConfig(
                input_dim=model_cfg.get("input_dim", len(self.config.data.features)),
                patch_length=model_cfg.get("patch_length", 16),
                stride=model_cfg.get("stride", 8),
                d_model=model_cfg.get("d_model", 128),
                nheads=model_cfg.get("nheads", 8),
                depth=model_cfg.get("depth", 4),
                dropout=model_cfg.get("dropout", 0.1),
            )
            return PatchTSTBackbone(config=patch_config)
        if model_type == "fusion":
            vocab_size = model_cfg.get("token_vocab_size") or self._vocab_size()
            if vocab_size is None:
                raise ValueError("Fusion model requires token_vocab_size or data.vocab_path")
            fusion_config = FusionConfig(
                continuous_dim=model_cfg.get("continuous_dim", len(self.config.data.features)),
                token_vocab_size=vocab_size,
                d_model=model_cfg.get("d_model", 128),
                nheads=model_cfg.get("nheads", 4),
                depth=model_cfg.get("depth", 2),
                dropout=model_cfg.get("dropout", 0.1),
                token_embed_dim=model_cfg.get("token_embed_dim", 64),
            )
            return FusionBackbone(config=fusion_config)
        raise NotImplementedError(f"Unsupported model type: {model_type}")

    def build_module(self, backbone: nn.Module) -> ForecastingModule:
        optimizer_params = OptimizerParams(
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )
        return ForecastingModule(backbone=backbone, horizon=self.config.data.horizon, optimizer_params=optimizer_params)

    def build_datamodule(self, frame: pd.DataFrame):
        builder = DataModuleBuilder(self.config.data)
        return builder.build_datamodule(frame)

    def generate_diagnostics(self, frame: pd.DataFrame) -> dict[str, object]:
        if self.config.model.get("type") == "fusion":
            return fusion_token_report(frame, self.config.data)
        return {}

    def run(self) -> pl.Trainer:
        frame = self.load_frame()
        datamodule = self.build_datamodule(frame)
        self._datamodule = datamodule
        backbone = self.build_backbone()
        module = self.build_module(backbone)

        trainer_cfg = self.config.trainer
        trainer_kwargs = dict(
            max_epochs=trainer_cfg.max_epochs,
            accelerator=trainer_cfg.accelerator,
            precision=trainer_cfg.precision,
            gradient_clip_val=trainer_cfg.gradient_clip_val,
            default_root_dir=str(Path(self.config.output_dir)),
        )
        if trainer_cfg.devices is not None:
            trainer_kwargs["devices"] = trainer_cfg.devices
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(module, datamodule=datamodule)
        self._report = self.generate_diagnostics(frame)
        return trainer

    @property
    def datamodule(self):
        return self._datamodule

    def test(self, trainer: pl.Trainer):
        if self._datamodule is None:
            raise RuntimeError("No datamodule available; run() must be called first")
        return trainer.test(datamodule=self._datamodule)

    @property
    def report(self) -> dict[str, object]:
        return self._report
