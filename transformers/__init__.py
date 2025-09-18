"""Transformer-based financial modeling toolkit."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("trading_project_transformers")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
