"""Logging utilities for the trading_transformers package."""

from __future__ import annotations

import logging
from typing import Optional, TextIO

LOGGER_NAME = "trading_transformers"


def configure_logging(level: int = logging.INFO, stream: Optional[TextIO] = None) -> None:
    """Activate package-wide logging with timestamped output.

    Use this instead of ``logging.basicConfig`` when running notebooks or CLIs::

        from trading_transformers.logging import configure_logging
        configure_logging(level=logging.DEBUG)

    Parameters
    ----------
    level:
        Desired log level for the ``trading_transformers`` logger hierarchy.
    stream:
        Optional stream object. Defaults to ``sys.stderr`` when omitted.
    """

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handler.setLevel(level)
        logger.addHandler(handler)
        logger.propagate = False
    else:
        for handler in logger.handlers:
            handler.setLevel(level)


__all__ = ["configure_logging", "LOGGER_NAME"]
