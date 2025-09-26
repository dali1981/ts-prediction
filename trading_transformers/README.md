# Transformers Package Overview

This package contains the initial scaffolding for the transformer-based financial modeling roadmap. It currently covers:

- `data`: catalog management, zip archive staging, basic resampling utilities, and a CLI (`python -m trading_transformers.data`).
- `features`: continuous bar feature generation and Brooks-style token emission (rule-based prototype).
- `tokenizers`: vocabulary helper for Brooks tokens plus analytics CLI (`python -m trading_transformers.tokenizers`).
- `models`: PatchTST-inspired backbone, reference PatchTST variant, Hugging Face `TimeSeriesTransformer` wrapper, a fusion encoder for continuous+token inputs, and a simplified Temporal Fusion Transformer.
- `training`: experiment configuration dataclasses, sliding-window dataset builder, PyTorch Lightning forecasting module, experiment runner, and CLI (`python -m trading_transformers.training`) with diagnostics export.
- Validation now logs additional forecast diagnostics inspired by `enhanced_eval.py`: RMSE, MAPE, bias, volatility ratio, directional accuracy, long/short precision, hit-rate, and correlation alongside the existing loss/MAE metrics.
- Reference implementations for comparison: [PatchTST (official PyTorch repo)](https://github.com/yuqinie98/PatchTST?utm_source=chatgpt.com) and [Temporal Fusion Transformer (Google Research)](https://github.com/google-research/google-research/tree/master/tft) offer the original paper baselines used to benchmark this codebase.
- `evaluation`: metrics/backtest helpers plus CLIs (`python -m trading_transformers.evaluation`, `python -m trading_transformers.backtest`).

Next steps include tokenizer refinement, fusion architectures, and richer backtesting/reporting as outlined in `PLAN.md`.

### Tokenizer Analytics

Run descriptive stats on generated Brooks tokens::

    python -m trading_transformers.tokenizers stats --input tokens.csv --column tokens --top-k 15

Generate a co-occurrence matrix for sequence-level analysis::

    python -m trading_transformers.tokenizers cooccurrence --input token_sequences.csv --sequence-column sequence --delimiter ' ' --output cooccurrence.csv


### Fusion Encoder

Example construction combining continuous OHLCV features with Brooks tokens::

    from trading_transformers.models import FusionBackbone, FusionConfig
    config = FusionConfig(continuous_dim=16, token_vocab_size=len(vocab.tokens))
    model = FusionBackbone(config)
    output = model(continuous_batch, token_batch)

The fusion encoder sums projected continuous features with embedded token representations before feeding them through a Transformer encoder, enabling multimodal experiments.


## Notebooks

- `notebooks/transformer_workflow.ipynb`: End-to-end synthetic walkthrough covering catalog, features, config, and training.
- `notebooks/tokenizer_analysis.ipynb`: Quickstart for inspecting Brooks tokens using the analytics helpers.

Example CLI call using the fusion config::

    python -m trading_transformers.training --config trading_transformers/configs/fusion.yaml --catalog notebooks/_tmp/catalog.json


### Diagnostics

Generate fusion token reports programmatically::

    from trading_transformers.evaluation import fusion_token_report
    report = fusion_token_report(frame, data_cfg)

The training CLI can persist diagnostics via `--diagnostics fusion_report.json`.

Example with explicit MPS accelerator::

    python -m trading_transformers.training --config trading_transformers/configs/fusion.yaml \
        --catalog notebooks/_tmp/catalog.json --accelerator mps --devices 1
