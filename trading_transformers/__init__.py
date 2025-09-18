"""Transformer-based financial modeling toolkit."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("trading-transformers-toolkit")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
