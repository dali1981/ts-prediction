"""Wrapper for Hugging Face TimeSeriesTransformer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
except ImportError:  # pragma: no cover
    TimeSeriesTransformerConfig = None
    TimeSeriesTransformerForPrediction = None

@dataclass(slots=True)
class HFTimeSeriesConfig:
    prediction_length: int
    context_length: int
    input_size: int
    extra: Dict[str, Any] = field(default_factory=dict)


class TimeSeriesTransformerWrapper:
    """Configurable wrapper around Hugging Face TimeSeriesTransformer."""

    def __init__(self, config: HFTimeSeriesConfig) -> None:
        cfg_dict = {
            "prediction_length": config.prediction_length,
            "context_length": config.context_length,
            "input_size": config.input_size,
        }
        cfg_dict.update(config.extra)
        if TimeSeriesTransformerConfig is None or TimeSeriesTransformerForPrediction is None:
            raise RuntimeError("Hugging Face transformers package is not available")
        self.config = TimeSeriesTransformerConfig(**cfg_dict)
        self.model = TimeSeriesTransformerForPrediction(self.config)

    def forward(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)

    def to(self, device: str) -> "TimeSeriesTransformerWrapper":
        self.model.to(device)
        return self
