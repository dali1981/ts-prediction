"""Diagnostic helpers for transformer experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, TYPE_CHECKING

import pandas as pd

from ..tokenizers import BrooksTokenVocabulary, TokenStats, compute_stats

if TYPE_CHECKING:
    from ..training.config import DataConfig


def _stats_to_dict(stats: TokenStats) -> Dict[str, object]:
    return {
        "total_tokens": stats.total_tokens,
        "unique_tokens": stats.unique_tokens,
        "entropy": stats.entropy,
        "top_tokens": stats.top_tokens,
    }


def fusion_token_report(frame: pd.DataFrame, data_cfg: "DataConfig") -> Dict[str, object]:
    """Generate simple diagnostics for fusion experiments."""
    report: Dict[str, object] = {}
    if not data_cfg.token_column:
        return report
    if data_cfg.token_column not in frame.columns:
        raise ValueError(f"Token column {data_cfg.token_column} missing from frame")

    tokens = frame[data_cfg.token_column].dropna().astype(str).tolist()
    if not tokens:
        return report

    stats = compute_stats(tokens)
    report["token_stats"] = _stats_to_dict(stats)

    sample = tokens[:20]
    report["sample_tokens"] = sample

    if data_cfg.vocab_path:
        vocab_path = Path(data_cfg.vocab_path)
        if vocab_path.exists():
            vocab = BrooksTokenVocabulary.from_json(vocab_path)
            report["vocab_size"] = len(vocab.tokens)

    return report
