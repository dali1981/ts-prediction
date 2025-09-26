"""Continuous feature engineering for OHLCV data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from ..logging import LOGGER_NAME


FeatureBuilder = Callable[[pd.DataFrame], pd.Series | pd.DataFrame]


@dataclass(slots=True)
class ContinuousFeatureConfig:
    """Configuration for continuous feature engineering."""

    lookback: int = 64
    include_volume: bool = True
    include_volatility: bool = True
    include_calendar: bool = True
    include_range: bool = True
    include_returns: bool = True
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: Optional[str] = "volume"
    timestamp_col: Optional[str] = "timestamp"
    extra_feature_builders: tuple[FeatureBuilder, ...] = field(default_factory=tuple)


class ContinuousFeatureBuilder:
    """Compute normalized continuous features for OHLCV frames."""

    def __init__(self, config: Optional[ContinuousFeatureConfig] = None) -> None:
        self.config = config or ContinuousFeatureConfig()
        self.log = logging.getLogger(LOGGER_NAME)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return ``frame`` augmented with price-derived indicators."""

        self._validate_required_columns(frame)
        out = frame.copy()

        if self.config.include_returns:
            self.log.debug("Adding return features")
            self._add_return_features(out)
        if self.config.include_range:
            self.log.debug("Adding range features")
            self._add_range_features(out)
        if self.config.include_volume:
            self.log.debug("Adding volume features")
            self._add_volume_features(out)
        if self.config.include_volatility:
            self.log.debug("Adding volatility features")
            self._add_volatility_features(out)
        if self.config.include_calendar:
            self.log.debug("Adding calendar features")
            self._add_calendar_features(out)

        self._apply_extra_builders(out)
        return out

    # ------------------------------------------------------------------ helpers --
    def _validate_required_columns(self, frame: pd.DataFrame) -> None:
        cfg = self.config
        required = {cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    def _series(self, frame: pd.DataFrame, col: str) -> pd.Series:
        return frame[col].astype(float)

    def _prev_close(self, close: pd.Series) -> pd.Series:
        return close.shift(1).replace(0.0, np.nan)

    def _add_return_features(self, out: pd.DataFrame) -> None:
        cfg = self.config
        close = self._series(out, cfg.close_col)
        prev_close = self._prev_close(close)

        out["close_return"] = close.diff()
        out["log_return"] = np.log(close / prev_close)

    def _add_range_features(self, out: pd.DataFrame) -> None:
        cfg = self.config
        high = self._series(out, cfg.high_col)
        low = self._series(out, cfg.low_col)
        close = self._series(out, cfg.close_col)
        prev_close = self._prev_close(close)

        spread = high - low
        out["hl_spread"] = spread
        out["hl_range_pct"] = spread / prev_close
        out["close_to_low"] = close - low
        out["close_location"] = (close - low) / spread.replace(0.0, np.nan)

    def _add_volume_features(self, out: pd.DataFrame) -> None:
        cfg = self.config
        volume_col = cfg.volume_col
        if not volume_col:
            raise ValueError("volume_col must be provided when include_volume=True")
        if volume_col not in out.columns:
            raise ValueError(f"Volume column '{volume_col}' missing from frame")

        volume = self._series(out, volume_col)
        out["volume"] = volume.astype(float)
        out["volume_change"] = volume.diff()

    def _add_volatility_features(self, out: pd.DataFrame) -> None:
        cfg = self.config
        close = self._series(out, cfg.close_col)
        prev_close = self._prev_close(close)
        returns = np.log(close / prev_close).fillna(0.0)
        out["rolling_volatility"] = returns.rolling(cfg.lookback).std()

    def _add_calendar_features(self, out: pd.DataFrame) -> None:
        cfg = self.config
        timestamp_col = cfg.timestamp_col
        if not timestamp_col:
            raise ValueError("timestamp_col must be provided when include_calendar=True")
        if timestamp_col not in out.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' missing from frame")

        time_index = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")
        day_of_week = time_index.dt.dayofweek
        out["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        out["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    def _apply_extra_builders(self, out: pd.DataFrame) -> None:
        for builder in self.config.extra_feature_builders:
            self.log.debug("Applying extra feature builder %s", getattr(builder, "__name__", repr(builder)))
            extra = builder(out)
            if isinstance(extra, pd.Series):
                name = extra.name or builder.__name__
                out[name] = extra
            elif isinstance(extra, pd.DataFrame):
                for col in extra.columns:
                    out[col] = extra[col]
            else:
                raise TypeError("extra_feature_builders must return a pandas Series or DataFrame")

    # -------------------------------------------------------------- normalization --
    @staticmethod
    def normalize(
        frame: pd.DataFrame,
        columns: Iterable[str],
        window: int,
        suffix: str = "_z",
    ) -> pd.DataFrame:
        """Return a copy of ``frame`` with rolling z-scores for the given columns."""

        normalized = frame.copy()
        logger = logging.getLogger(LOGGER_NAME)
        for col in columns:
            if col not in normalized.columns:
                raise KeyError(f"Column '{col}' not found for normalization")
            logger.debug("Normalizing column %s with window %d", col, window)
            series = normalized[col].astype(float)
            rolling = series.rolling(window)
            std = rolling.std().replace(0.0, np.nan)
            normalized[f"{col}{suffix}"] = (series - rolling.mean()) / std
        return normalized



def generate_features(frame, config=None):
    """Helper to mirror legacy API by returning engineered features."""
    builder = ContinuousFeatureBuilder(config)
    return builder.transform(frame)


__all__ = [
    "ContinuousFeatureBuilder",
    "ContinuousFeatureConfig",
    "generate_features",
]
