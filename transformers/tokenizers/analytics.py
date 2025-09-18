"""Analytics helpers for Brooks-style token sequences."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(slots=True)
class TokenStats:
    total_tokens: int
    unique_tokens: int
    top_tokens: List[Tuple[str, int]]
    entropy: float


def compute_stats(sequences: Iterable[str], top_k: int = 20) -> TokenStats:
    counter = Counter(sequences)
    total = sum(counter.values())
    if total == 0:
        raise ValueError("No tokens provided")
    unique = len(counter)
    top = counter.most_common(top_k)
    freq_series = pd.Series(counter, dtype=float)
    probs = freq_series / total
    safe_probs = probs.replace(0.0, np.nan)
    entropy = float(-(safe_probs * np.log2(safe_probs)).sum(skipna=True))
    return TokenStats(
        total_tokens=total,
        unique_tokens=unique,
        top_tokens=top,
        entropy=entropy,
    )


def cooccurrence_matrix(sequences: Iterable[List[str]]) -> pd.DataFrame:
    """Return co-occurrence counts for tokens appearing within the same window."""
    rows: Dict[Tuple[str, str], int] = {}
    for seq in sequences:
        unique_tokens = set(seq)
        for token in unique_tokens:
            rows[(token, token)] = rows.get((token, token), 0) + 1
        tokens = list(unique_tokens)
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                pair = tuple(sorted((tokens[i], tokens[j])))
                rows[pair] = rows.get(pair, 0) + 1
    if not rows:
        return pd.DataFrame()
    data = {(a, b): count for (a, b), count in rows.items()}
    frame = pd.Series(data).rename('count').reset_index()
    frame.columns = ['token_a', 'token_b', 'count']
    pivot = frame.pivot_table(index='token_a', columns='token_b', values='count', fill_value=0)
    return pivot
