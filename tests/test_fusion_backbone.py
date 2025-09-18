import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from trading_transformers.models import FusionBackbone, FusionConfig


def test_fusion_backbone_forward():
    config = FusionConfig(continuous_dim=5, token_vocab_size=10, d_model=32, nheads=4, depth=1)
    model = FusionBackbone(config)
    batch = 4
    lookback = 16
    continuous = torch.randn(batch, lookback, config.continuous_dim)
    tokens = torch.randint(0, config.token_vocab_size, (batch, lookback))
    output = model(continuous, tokens)
    assert output.shape == (batch, lookback, config.d_model)
