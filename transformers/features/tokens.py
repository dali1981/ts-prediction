"""Rule-based token emission for Brooks-style price action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass(slots=True)
class BrooksTokenizerConfig:
    body_bins: int = 5
    tail_bins: int = 3
    regime_window: int = 20


class BrooksTokenizer:
    """Produce coarse-grained categorical tokens for each bar."""

    def __init__(self, config: Optional[BrooksTokenizerConfig] = None) -> None:
        self.config = config or BrooksTokenizerConfig()

    def fit(self, frame: pd.DataFrame) -> "BrooksTokenizer":
        return self

    def transform(self, frame: pd.DataFrame) -> List[str]:
        required = {"open", "high", "low", "close"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

        tokens: List[str] = []
        highs = frame["high"].astype(float)
        lows = frame["low"].astype(float)
        closes = frame["close"].astype(float)
        opens = frame["open"].astype(float)

        bodies = closes - opens
        ranges = (highs - lows).replace(0.0, pd.NA)
        body_ratio = (bodies.abs() / ranges).fillna(0.0)

        body_bins = pd.qcut(body_ratio, self.config.body_bins, labels=False, duplicates="drop")
        tail_upper = ((highs - closes).abs() / ranges).fillna(0.0)
        tail_bins = pd.qcut(tail_upper, self.config.tail_bins, labels=False, duplicates="drop")

        regime = closes.rolling(self.config.regime_window).mean() - closes
        regime_label = regime.apply(lambda x: "trend_down" if x > 0 else "trend_up")

        direction = bodies.apply(lambda x: "bull" if x > 0 else ("bear" if x < 0 else "doji"))

        for idx in range(len(frame)):
            parts = [direction.iloc[idx], f"body{int(body_bins.iloc[idx])}" if not pd.isna(body_bins.iloc[idx]) else "bodyNA"]
            tail = tail_bins.iloc[idx]
            parts.append(f"tail{int(tail)}" if not pd.isna(tail) else "tailNA")
            parts.append(regime_label.iloc[idx] if isinstance(regime_label.iloc[idx], str) else "regimeNA")
            tokens.append("|".join(parts))

        return tokens
