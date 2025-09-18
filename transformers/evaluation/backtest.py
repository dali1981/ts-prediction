"""Lightweight backtesting helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(slots=True)
class Backtester:
    transaction_cost: float = 0.0

    def run(self, signals: pd.Series, returns: pd.Series) -> Dict[str, float]:
        aligned = signals.index.intersection(returns.index)
        signals = signals.loc[aligned]
        returns = returns.loc[aligned]
        pnl = signals.shift(1).fillna(0.0) * returns
        costs = np.abs(signals.diff()).fillna(0.0) * self.transaction_cost
        net = pnl - costs
        equity = net.cumsum()
        sharpe = net.mean() / net.std(ddof=1) if net.std(ddof=1) else 0.0
        drawdown = (equity.cummax() - equity).max()
        return {
            "total_return": float(net.sum()),
            "sharpe": float(sharpe),
            "max_drawdown": float(drawdown),
        }
