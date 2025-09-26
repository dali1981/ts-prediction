# Trading Transformers Data Loading

This note explains how market data is staged, normalised, and handed off to modeling code. It deliberately separates the **data interface** from downstream **model families** so that shared loaders can feed both transformer-based workflows and alternative models (Chronos, LSTM, CNN, etc.).

## System Overview

1. **Catalog registration** — index raw CSV/Parquet files or archives with `DataCatalog`.
2. **Frame loading & staging** — pull a `pd.DataFrame` via catalog helpers, optionally extract zip archives, and apply resampling.
3. **Feature enrichment** — add engineered columns (continuous features, rule-based tokens) using utilities under `trading_transformers.features`.
4. **Window generation** — slice supervised windows with `WindowGenerator` and friends, producing tensors for any sequence model.
5. **Model execution** — pass the resulting loaders to transformers, Chronos, CNN/LSTM baselines, or analytics notebooks.

Every step before (5) is **model-agnostic**; tokenisation and Lightning helpers are optional layers that only transformers rely on.

## Catalog & Source Registration

Defined in `trading_transformers.data.catalog`:

- `DataSource` tracks individual CSV/Parquet assets.
- `ZipArchive` points to compressed bundles and extracts them into a cache folder.

```python
from trading_transformers.data import DataCatalog, DataSource, Resampler

catalog = DataCatalog(root="/data/catalog")
catalog.register_source(DataSource(
    name="ethusdt_1min",
    path=catalog.root / "ethusdt.csv",
    fmt="csv",
    frequency="1min",
))
frame = catalog.load("ethusdt_1min", parse_dates=["timestamp"], index_col="timestamp")
frame_15m = Resampler.for_ohlcv("15T").apply(frame)
```

CLI support (`python -m trading_transformers.data`) mirrors the same operations for reproducible catalog files consumed by multiple projects.

## Feature Engineering Layers

Feature builders live in `trading_transformers.features` and can be composed depending on the target model:

- **Continuous features** (`ContinuousFeatureBuilder`) — constructs price-driven indicators (returns, spreads, calendar signals). Use `ContinuousFeatureBuilder.normalize(...)` later if you need rolling z-scores or other scaling.
- **Token features** (`tokens.py`) — rule-based Brooks-style symbols for transformer fusion experiments.

Tip: enable debug logs when stepping through feature preparation with::

    from trading_transformers.logging import configure_logging
    configure_logging(level=logging.DEBUG)

Keep the raw catalog output untouched; derive model-specific views by copying and augmenting frames in notebooks or preprocessing scripts.

## Window Generation

`trading_transformers.training.datamodule` provides reusable windowing logic:

- `WindowGenerator` (core): validates schemas, encodes optional token columns, and returns numpy arrays.
- `SlidingWindowDataset` / `DataModuleBuilder`: wrap arrays as PyTorch `Dataset`/Lightning `DataModule` objects.
- Set `DataConfig.include_future_features=True` when working with decoder-style models (e.g., the Temporal Fusion Transformer). This ensures the generator emits an additional `future` tensor of shape `(batch, horizon, features)` so backbones that require known future covariates receive the right structure.

Because `WindowGenerator.generate` returns `(continuous, targets, tokens?)`, non-transformer models can ignore the token output entirely:

```python
cont, targets, _ = WindowGenerator(cfg_without_tokens).generate(feature_frame)
```

For models that prefer flat tensors (Chronos, classic RNNs), simply reshape the `continuous` block or bypass the Lightning module in favour of custom loaders.

## Separation of Responsibilities

### Transformers

- Expect both continuous tensors and (optionally) encoded token sequences.
- Leverage `DataModuleBuilder` for Lightning integration, diagnostics, and fusion reporting.
- Rely on vocab files when discretising Brooks tokens.

### Other Models (Chronos, LSTM, CNN, etc.)

- Consume the same continuous feature windows.
- May skip token logic entirely or supply their own embedding scheme.
- Can reuse `WindowGenerator` + `SlidingWindowDataset` directly, or export numpy arrays into bespoke training pipelines.

The guiding rule: **data loaders stay generic**, while model-specific preprocessing (token embeddings, normalisation strategy, batching rules) lives closer to the estimator.

## Current Constraints

- Catalog JSON currently stores paths verbatim; keep execution within the same directory tree until the loader is updated to rebase relative paths.
- Sliding-window generation drops rows with NaNs before stacking; clean or backfill mandatory columns before windowing.
- Token handling assumes integer indices when no vocabulary is provided. Future revisions will auto-encode strings or require an explicit vocab.

### Matplotlib Cache

Test runs or notebooks that import Matplotlib inside the repo may emit warnings about unwritable cache directories (e.g., `~/.matplotlib`). Set a local cache path before running plots:

```
export MPLCONFIGDIR="$(pwd)/.mpl-cache"
mkdir -p "$MPLCONFIGDIR"
```

Keeping this snippet in your test harness or environment activation script avoids repeated cache rebuilds and silences the warning.

By keeping these limits in mind, the shared loading stack remains robust for the transformer roadmap and any additional forecasting baselines.
