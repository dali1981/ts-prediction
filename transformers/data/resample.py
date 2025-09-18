"""Resampling utilities for financial time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

OHLC_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


@dataclass(slots=True)
class Resampler:
    rule: str
    agg_map: Optional[Dict[str, str]] = None

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Resample the frame using pandas frequency rules."""
        if "timestamp" in frame.columns:
            index = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        else:
            index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        grouped = frame.copy()
        grouped.index = index
        agg_map = self.agg_map or OHLC_AGG
        result = grouped.resample(self.rule).agg(agg_map)
        return result.dropna(how="all")

    @classmethod
    def for_ohlcv(cls, rule: str) -> "Resampler":
        return cls(rule=rule, agg_map=OHLC_AGG)
