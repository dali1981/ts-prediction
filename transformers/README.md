# Transformers Package Overview

This package contains the initial scaffolding for the transformer-based financial modeling roadmap. It currently covers:

- `data`: catalog management, zip archive staging, basic resampling utilities, and a CLI (`python -m transformers.data`).
- `features`: continuous bar feature generation and Brooks-style token emission (rule-based prototype).
- `tokenizers`: vocabulary helper for Brooks tokens plus analytics CLI (`python -m transformers.tokenizers`).
- `models`: PatchTST-inspired backbone, Hugging Face `TimeSeriesTransformer` wrapper, and a fusion encoder for continuous+token inputs.
- `training`: experiment configuration dataclasses, sliding-window dataset builder, PyTorch Lightning forecasting module, experiment runner, and CLI (`python -m transformers.training`) with diagnostics export.
- `evaluation`: metrics/backtest helpers plus CLIs (`python -m transformers.evaluation`, `python -m transformers.backtest`).

Next steps include tokenizer refinement, fusion architectures, and richer backtesting/reporting as outlined in `PLAN.md`.

### Tokenizer Analytics

Run descriptive stats on generated Brooks tokens::

    python -m transformers.tokenizers stats --input tokens.csv --column tokens --top-k 15

Generate a co-occurrence matrix for sequence-level analysis::

    python -m transformers.tokenizers cooccurrence --input token_sequences.csv --sequence-column sequence --delimiter ' ' --output cooccurrence.csv


### Fusion Encoder

Example construction combining continuous OHLCV features with Brooks tokens::

    from transformers.models import FusionBackbone, FusionConfig
    config = FusionConfig(continuous_dim=16, token_vocab_size=len(vocab.tokens))
    model = FusionBackbone(config)
    output = model(continuous_batch, token_batch)

The fusion encoder sums projected continuous features with embedded token representations before feeding them through a Transformer encoder, enabling multimodal experiments.


## Notebooks

- `notebooks/transformer_workflow.ipynb`: End-to-end synthetic walkthrough covering catalog, features, config, and training.
- `notebooks/tokenizer_analysis.ipynb`: Quickstart for inspecting Brooks tokens using the analytics helpers.

Example CLI call using the fusion config::

    python -m transformers.training --config transformers/configs/fusion.yaml --catalog notebooks/_tmp/catalog.json


### Diagnostics

Generate fusion token reports programmatically::

    from transformers.evaluation import fusion_token_report
    report = fusion_token_report(frame, data_cfg)

The training CLI can persist diagnostics via `--diagnostics fusion_report.json`.

Example with explicit MPS accelerator::

    python -m transformers.training --config transformers/configs/fusion.yaml \
        --catalog notebooks/_tmp/catalog.json --accelerator mps --devices 1

