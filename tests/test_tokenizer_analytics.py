import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

pd = pytest.importorskip("pandas")

from transformers.features.tokens import BrooksTokenizer
from transformers.tokenizers import BrooksTokenVocabulary, compute_stats, cooccurrence_matrix


def test_compute_stats_and_vocab():
    frame = pd.DataFrame({
        "open": [100, 101, 102, 101, 100],
        "high": [101, 102, 103, 102, 101],
        "low": [99, 100, 101, 100, 99],
        "close": [100.5, 101.5, 102.5, 101.0, 100.0],
    })
    tokenizer = BrooksTokenizer()
    tokens = tokenizer.transform(frame)

    stats = compute_stats(tokens, top_k=3)
    assert stats.total_tokens == len(tokens)
    assert stats.unique_tokens <= stats.total_tokens
    assert len(stats.top_tokens) <= 3
    assert stats.entropy >= 0

    vocab = BrooksTokenVocabulary.from_sequences(tokens)
    assert vocab.encode(tokens[0]) >= 0

    sequences = [tokens[:3], tokens[2:]]
    matrix = cooccurrence_matrix(sequences)
    assert not matrix.empty
    assert matrix.values.diagonal().sum() > 0
