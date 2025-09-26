"""Model definitions and wrappers."""

from .patchtst import PatchTSTBackbone, PatchTSTConfig
from .patchtst_reference import PatchTSTReference, PatchTSTReferenceConfig
from .fusion import FusionBackbone, FusionConfig
from .tft import TemporalFusionTransformerBackbone, TemporalFusionTransformerConfig

__all__ = [
    "PatchTSTBackbone",
    "PatchTSTConfig",
    "PatchTSTReference",
    "PatchTSTReferenceConfig",
    "TemporalFusionTransformerBackbone",
    "TemporalFusionTransformerConfig",
    "FusionBackbone",
    "FusionConfig",
]

try:  # pragma: no cover - optional Hugging Face dependency
    from .hf_time_series import TimeSeriesTransformerWrapper
except (RuntimeError, ImportError):
    TimeSeriesTransformerWrapper = None  # type: ignore[assignment]
else:
    __all__.append("TimeSeriesTransformerWrapper")
