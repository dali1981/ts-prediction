from numpy.random import default_rng
import pandas as pd
import torch

from trading_transformers.features import ContinuousFeatureBuilder, ContinuousFeatureConfig
from trading_transformers.models import (
    PatchTSTBackbone,
    PatchTSTConfig,
    PatchTSTReference,
    PatchTSTReferenceConfig,
    TemporalFusionTransformerBackbone,
    TemporalFusionTransformerConfig,
)
from trading_transformers.training.config import DataConfig
from trading_transformers.training.datamodule import WindowGenerator
from trading_transformers.training.lightning import ForecastingModule, OptimizerParams


def _make_sample_frame(rows: int = 128) -> pd.DataFrame:
    rng = default_rng(7)
    index = pd.date_range("2024-01-01", periods=rows, freq="min")
    prices = 100 + rng.normal(0, 0.5, size=rows).cumsum()
    highs = prices + rng.uniform(0.1, 0.5, size=rows)
    lows = prices - rng.uniform(0.1, 0.5, size=rows)
    closes = prices + rng.uniform(-0.2, 0.2, size=rows)
    volume = rng.uniform(1000, 1500, size=rows)
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": prices,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        }
    )


def _build_feature_frame(rows: int = 128) -> pd.DataFrame:
    frame = _make_sample_frame(rows)
    builder = ContinuousFeatureBuilder(ContinuousFeatureConfig(lookback=10))
    features = builder.transform(frame)
    return features.dropna().reset_index(drop=True)


def _common_config(include_future: bool) -> DataConfig:
    return DataConfig(
        source="dummy",
        features=[
            "log_return",
            "hl_spread",
            "hl_range_pct",
            "close_location",
            "volume_change",
            "rolling_volatility",
        ],
        target="close_return",
        lookback=16,
        horizon=4,
        batch_size=8,
        include_future_features=include_future,
    )


def _prepare_batch(cfg: DataConfig, frame: pd.DataFrame, include_future: bool):
    generator = WindowGenerator(cfg)
    continuous, targets, _, future = generator.generate(frame, include_future=include_future)
    batch_features = torch.tensor(continuous[: cfg.batch_size], dtype=torch.float32)
    batch_targets = torch.tensor(targets[: cfg.batch_size], dtype=torch.float32)
    batch_future = (
        torch.tensor(future[: cfg.batch_size], dtype=torch.float32) if future is not None else None
    )
    return batch_features, batch_targets, batch_future


def test_window_generator_shapes():
    feature_frame = _build_feature_frame()
    cfg = _common_config(include_future=False)
    generator = WindowGenerator(cfg)
    continuous, targets, tokens, future = generator.generate(feature_frame)

    assert continuous.shape[0] > 0
    assert continuous.shape[1] == cfg.lookback
    assert continuous.shape[2] == len(cfg.features)
    assert targets.shape[1] == cfg.horizon
    assert tokens is None
    assert future is None


def test_forecasting_metrics_are_finite():
    frame = _build_feature_frame()
    cfg = _common_config(include_future=False)
    batch_features, batch_targets, _ = _prepare_batch(cfg, frame, include_future=False)

    backbone = PatchTSTBackbone(
        PatchTSTConfig(
            input_dim=len(cfg.features),
            patch_length=4,
            stride=2,
            d_model=32,
            nheads=4,
            depth=1,
            dropout=0.1,
        )
    )
    module = ForecastingModule(
        backbone=backbone,
        horizon=cfg.horizon,
        optimizer_params=OptimizerParams(lr=1e-3, weight_decay=0.0),
    )
    module.log = lambda *args, **kwargs: None

    module.on_validation_epoch_start()
    module.validation_step(({"continuous": batch_features}, batch_targets), 0)
    module.on_validation_epoch_end()

    metrics = module._additional_metrics(batch_targets, batch_targets)
    for value in metrics.values():
        assert torch.isfinite(value).all()


def test_patchtst_reference_forward():
    frame = _build_feature_frame()
    cfg = _common_config(include_future=False)
    batch_features, _, _ = _prepare_batch(cfg, frame, include_future=False)

    backbone = PatchTSTReference(
        PatchTSTReferenceConfig(
            input_dim=len(cfg.features),
            patch_length=4,
            stride=2,
            d_model=64,
            nheads=4,
            depth=2,
            dropout=0.1,
        )
    )
    output = backbone(batch_features)
    assert output.shape[0] == batch_features.shape[0]
    assert output.shape[-1] == backbone.config.d_model


def test_tft_backbone_metrics():
    frame = _build_feature_frame()
    cfg = _common_config(include_future=True)
    batch_features, batch_targets, batch_future = _prepare_batch(cfg, frame, include_future=True)

    backbone = TemporalFusionTransformerBackbone(
        TemporalFusionTransformerConfig(
            input_dim=len(cfg.features),
            d_model=64,
            nheads=4,
            depth=1,
            dropout=0.1,
        )
    )
    module = ForecastingModule(
        backbone=backbone,
        horizon=cfg.horizon,
        optimizer_params=OptimizerParams(lr=1e-3, weight_decay=0.0),
    )
    module.log = lambda *args, **kwargs: None

    module.on_validation_epoch_start()
    module.validation_step(
        ({"continuous": batch_features, "future": batch_future}, batch_targets), 0
    )
    module.on_validation_epoch_end()

    metrics = module._additional_metrics(batch_targets, batch_targets)
    for value in metrics.values():
        assert torch.isfinite(value).all()
