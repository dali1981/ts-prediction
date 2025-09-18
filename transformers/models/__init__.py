"""Model definitions and wrappers."""

from .patchtst import PatchTSTBackbone
from .fusion import FusionBackbone, FusionConfig

__all__ = ["PatchTSTBackbone", "FusionBackbone", "FusionConfig"]

try:  # pragma: no cover - optional Hugging Face dependency
    from .hf_time_series import TimeSeriesTransformerWrapper
except (RuntimeError, ImportError):
    TimeSeriesTransformerWrapper = None  # type: ignore[assignment]
else:
    __all__.append("TimeSeriesTransformerWrapper")
