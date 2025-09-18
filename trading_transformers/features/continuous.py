"""Continuous feature engineering for OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ContinuousFeatureConfig:
    lookback: int = 64
    include_volume: bool = True
    include_volatility: bool = True
    calendar_features: bool = True


class ContinuousFeatureBuilder:
    """Compute normalized continuous features for OHLCV frames."""

    def __init__(self, config: Optional[ContinuousFeatureConfig] = None) -> None:
        self.config = config or ContinuousFeatureConfig()

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Augment input OHLCV frame with engineered features."""
        required = {"open", "high", "low", "close"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

        out = frame.copy()
        out["log_return"] = np.log(out["close"].astype(float) / out["close"].astype(float).shift(1))
        out["hl_range"] = (out["high"].astype(float) - out["low"].astype(float)) / out["close"].astype(float).shift(1)
        out["close_location"] = (out["close"].astype(float) - out["low"].astype(float)) / (
            (out["high"].astype(float) - out["low"].astype(float)).replace(0.0, np.nan)
        )

        if self.config.include_volume and "volume" in out.columns:
            vol = out["volume"].astype(float)
            out["volume_z"] = (vol - vol.rolling(self.config.lookback).mean()) / vol.rolling(self.config.lookback).std()

        if self.config.include_volatility:
            returns = out["log_return"].fillna(0.0)
            out["rv"] = returns.rolling(self.config.lookback).std() * np.sqrt(self.config.lookback)

        if self.config.calendar_features and "timestamp" in out.columns:
            time_index = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
            out["dow_sin"] = np.sin(2 * np.pi * time_index.dt.dayofweek / 7)
            out["dow_cos"] = np.cos(2 * np.pi * time_index.dt.dayofweek / 7)

        return out



def generate_features(frame, config=None):
    """Helper to mirror legacy API by returning engineered features."""
    builder = ContinuousFeatureBuilder(config)
    return builder.transform(frame)


__all__ = ["ContinuousFeatureBuilder", "ContinuousFeatureConfig", "generate_features"]
