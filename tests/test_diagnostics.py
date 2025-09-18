import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from transformers.evaluation.diagnostics import fusion_token_report
from types import SimpleNamespace


def test_fusion_token_report():
    frame = pd.DataFrame({
        "open": [1, 2, 3],
        "close": [1, 2, 3],
        "brooks_token": ["bull|body1|tail0|trend_up", "bear|body2|tail1|trend_down", "bull|body1|tail0|trend_up"],
    })

    cfg = SimpleNamespace(token_column="brooks_token", vocab_path=None)

    report = fusion_token_report(frame, cfg)
    assert "token_stats" in report
    assert report["token_stats"]["total_tokens"] == len(frame)
    assert len(report["sample_tokens"]) <= len(frame)
