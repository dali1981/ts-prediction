"""Transformer-based financial modeling toolkit."""

from importlib.metadata import PackageNotFoundError, version

from .logging import configure_logging

try:
    __version__ = version("trading-transformers-toolkit")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__", "configure_logging"]
