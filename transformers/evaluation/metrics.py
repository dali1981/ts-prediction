"""Performance metrics for financial models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(slots=True)
class MetricsBundle:
    """Calculate common predictive and trading metrics."""

    def regression(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        error = y_pred - y_true
        return {
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt((error ** 2).mean())),
        }

    def directional(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        sign_true = np.sign(y_true)
        sign_pred = np.sign(y_pred)
        accuracy = (sign_true == sign_pred).mean()
        return {"hit_rate": float(accuracy)}

    def sharpe(self, returns: pd.Series, risk_free: float = 0.0) -> Dict[str, float]:
        excess = returns - risk_free
        mean = excess.mean()
        std = excess.std(ddof=1)
        sharpe = mean / std if std else 0.0
        return {"sharpe": float(sharpe)}
