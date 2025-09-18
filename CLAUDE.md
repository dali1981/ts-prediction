# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **transformer-based financial time-series modeling toolkit** for forecasting and trading signal generation on OHLCV data. The project implements:

1. **Data ingestion** via DataCatalog with support for zip archives and CSV files
2. **Two feature representations**: continuous normalized bars and Brooks-style price-action tokens
3. **Transformer models**: PatchTST baseline, HuggingFace TimeSeriesTransformer, and fusion architectures
4. **Evaluation framework** with walk-forward backtesting and trading KPIs

## Development Commands

### Python Environment
- Use `uv` for all Python operations:
  - `uv sync` - Install dependencies
  - `uv add <package>` - Add new dependencies
  - `uv run <command>` - Run Python commands

### Testing
- `uv run pytest` - Run all tests
- `uv run pytest tests/test_<module>.py` - Run specific test module

### Core CLI Commands
- **Training**: `uv run python -m trading_transformers.training --config configs/patchtst.yaml --catalog data/catalog.json`
- **Data management**: `uv run python -m trading_transformers.data init --root data --zip-dir data/archives`
- **Token analytics**: `uv run python -m trading_transformers.tokenizers stats --input tokens.csv --column tokens`
- **Backtesting**: `uv run python -m trading_transformers.evaluation backtest --config configs/backtest.yaml`

## Architecture

### Package Structure
```
trading_transformers/
├── data/           # DataCatalog, zip extraction, resampling utilities
├── features/       # Continuous bar features + Brooks tokenizer
├── tokenizers/     # Token vocabulary and analytics
├── models/         # PatchTST, TimeSeriesTransformer, FusionBackbone
├── training/       # PyTorch Lightning experiment runner
├── evaluation/     # Metrics, backtesting, walk-forward validation
└── configs/        # YAML experiment configurations
```

### Key Components

**DataCatalog**: Manages datasets with schemas, frequency metadata, and caching. Handles zip archives in `data/` directory and provides unified access to time-series data.

**Brooks Tokenizer**: Rule-based detector for price-action patterns (trend/range regime, swing levels, breakouts). Converts OHLCV bars to categorical tokens for sequence modeling.

**FusionBackbone**: Multimodal architecture combining continuous features with token embeddings via cross-attention, enabling experiments with both data representations.

**Experiment Framework**: PyTorch Lightning-based training with configurable data splits, loss functions, and evaluation metrics. Supports sliding-window datasets and walk-forward validation.

### Configuration Files
- `configs/patchtst.yaml` - PatchTST continuous transformer baseline
- `configs/fusion.yaml` - Multimodal continuous + token model
- `configs/backtest.yaml` - Backtesting and evaluation settings
- `configs/eval.yaml` - Evaluation metrics configuration

## Important Notes

- All CLI tools support both module execution (`python -m trading_transformers.training`) and direct execution
- Configuration uses YAML format with experiment configs, data specs, and model hyperparameters
- The fusion architecture requires both continuous features and token vocabularies
- Backtesting uses walk-forward splits with purged/embargoed cross-validation
- Data catalog must be initialized before running experiments