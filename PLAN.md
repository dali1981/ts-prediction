# Transformer-based Financial Time-Series Roadmap

## 1. Objectives & Scope
- Deliver a transformer toolkit for forecasting and trading signal generation on intraday/daily OHLCV data, with the option to extend to limit-order-book (LOB) streams.
- Support two data representations: (a) normalized continuous bars, (b) discretized "price-action tokens" inspired by Al Brooks patterns.
- Evaluate models on predictive accuracy and trading KPIs (Sharpe, max drawdown, turnover-adjusted P&L) via walk-forward backtests.

## 2. Datasets & Ingestion
- **Primary sources**: user-supplied datasets located under `@data` (validate mount/symlink) or `models/data` (zip archives), existing OHLCV parquet/CSV files in `/data/ohlcv`, plus optional macro factors and alternative data (news sentiment, volatility indices).
- **Ingestion tasks**:
  - Implement a `DataCatalog` wrapper with schemas, frequency metadata, and caching.
  - Provide resampling utilities (1m → 5m → 30m → 1D) with holiday-aware calendars.
  - Handle missing bars, corporate actions, and volume anomalies via forward/back-fill and filters.
  - Detect and unpack zip archives in `models/data`, validate checksums, and stage extracted files in a reproducible data lake hierarchy.
- **LOB extension (optional)**: parse order-book snapshots to 100-level tensors; align with trades for delta features.

## 3. Feature Engineering
- **Continuous bar features**:
  - Compute log-returns, high-low ranges, relative close position, rolling volatility z-scores, volume imbalance, realized volatility windows.
  - Add calendar embeddings (minute-of-day, day-of-week, month) using Time2Vec/Fourier features.
  - Optional exogenous features (macro, sentiment embeddings) aligned via left-join and forward-fill.
- **Brooks-style tokenizer**:
  - Implement rule-based detectors for primitives (trend/range regime, HH/HL/LH/LL swings, breakout attempts, pullback depth, measured-move status, magnet proximity).
  - Map each bar to a composite categorical token (e.g., `[trend=up, bar=bull_long, swing=HL, context=range, event=none]`).
  - Provide vocabulary builder, frequency analysis, entropy stats; export sequences as integer tensors.
- **Learned tokenizer (stretch)**:
  - Train VQ-VAE or TimeVQVAE on normalized candle patches to learn codebooks; expose encoder for inference-time token emission.

## 4. Modeling Architecture (PyTorch / HuggingFace)
- **Baseline (continuous)**:
  - Adapt PatchTST (ICLR'23) architecture in PyTorch: instance normalization, patch size hyperparameter, shared projection head per horizon.
  - Alternative: Hugging Face `TimeSeriesTransformer` for multi-horizon forecasting with probabilistic head (quantile/Poisson).
- **Brooks token model**:
  - Use Hugging Face `AutoModelForMaskedLM` with a `BertForMaskedLM` backbone on tokenized sequences for self-supervised pretraining.
  - Fine-tune `AutoModelForSequenceClassification` (Transformer encoder) on directional/volatility/Regime labels; optionally add CRF decoding for pattern tagging.
- **Fusion model**:
  - Concatenate continuous embeddings with token embeddings via cross-attention adapter (e.g., two-tower encoders feeding a gating layer).
  - Optional multimodal branch for news embeddings (FinBERT) fused via transformer encoder with shared positional timeline.
- **Microstructure module (optional)**:
  - Implement TransLOB-style model: causal CNN feature extractor + transformer encoder for order-book sequences.

## 5. Training Pipeline
- Structure experiments with PyTorch Lightning (or Lightning Fabric) for consistent training loops, mixed precision, and checkpointing.
- Implement configurable `ExperimentConfig` (YAML/JSON) controlling data splits, features, model hyperparameters, loss functions, and evaluation metrics.
- Provide dataloaders for:
  - Sliding-window supervised datasets (lookback `L`, forecast horizon `H`).
  - Masked LM tasks (random token masking) with dynamic batching by sequence length.
- Losses & heads:
  - Regression: quantile loss, mean absolute scaled error (MASE), tilt/Huber as needed.
  - Classification: focal loss or cross-entropy with label smoothing for imbalanced targets.
  - Reinforcement-style head (stretch): policy gradient or Sharpe maximization surrogate.
- Regularization: dropout, stochastic weight averaging, early stopping on validation Sharpe, temporal mixup, and feature noise injection.

## 6. Evaluation & Backtesting
- Establish rolling walk-forward splits (train N days → validate M days → test K days) with purged/embargoed cross-validation.
- Metrics: directional accuracy, precision/recall on up/down bins, RMSE/MAPE for regression, portfolio KPIs (annualized Sharpe, Sortino, Calmar, turnover, hit-rate).
- Build a lightweight backtester:
  - Signal → position sizing (Kelly fraction, volatility targeting).
  - Transaction cost model (spread + impact) and slippage simulation.
  - Reports with equity curves, drawdown plots, feature importance (attention, SHAP/Integrated Gradients).
- Logging & tracking: integrate with Weights & Biases or MLflow for experiment metadata, artifact storage, and hyperparameter sweeps.

## 7. Engineering & Infrastructure
- Package code into `models/transformers/` with modules `data`, `features`, `tokenizers`, `models`, `training`, `evaluation`.
- Implement unit/pytest suites for token logic, dataset slicing, and model forward passes (with small synthetic data).
- Provide CLI entrypoints:
  - `python -m transformers.train --config configs/patchtst.yaml`
  - `python -m transformers.tokenize --input data/raw --output data/tokens`
  - `python -m transformers.backtest --config configs/backtest.yaml`
- Create notebooks or Markdown playbooks demonstrating:
  - Exploratory analysis & token vocabulary stats.
  - Training PatchTST baseline.
  - Fine-tuning Hugging Face `TimeSeriesTransformer`.
- Set up Dockerfile or conda environment with pinned versions (`pytorch>=2.2`, `pytorch-lightning`, `transformers>=4.38`, `tslearn`, `vector-quantize-pytorch`).

## 8. Research References & Further Reading
- Nie et al., "PatchTST: Transformer for Time Series Forecasting" (ICLR 2023).
- Liu et al., "iTransformer: Inverted Transformations for Time Series Forecasting" (ICLR 2024).
- Lim et al., "Temporal Fusion Transformers" (NeurIPS 2019).
- Borovykh et al., "Time Series Transformer"
- Tran et al., "TransLOB" for limit order book forecasting.
- TOTEM / TimeVQVAE for discrete time-series tokenization.
- Stock2Vec / asset embedding literature for cross-sectional features.
- Hugging Face `TimeSeriesTransformer` documentation and tutorials.

## 9. Execution Timeline (suggested)
- **Week 1**: Data audit, ingestion pipelines, continuous feature engineering, baseline linear/GBDT benchmarks.
- **Week 2**: Implement PatchTST continuous transformer, set up Lightning training + first walk-forward results.
- **Week 3**: Build Brooks tokenizer, generate token datasets, run masked LM pretraining & classification fine-tune.
- **Week 4**: Fusion experiments, hyperparameter sweeps, full backtesting, reporting, and deployment packaging.
- **Week 5+ (stretch)**: Learned VQ tokenizers, multimodal news fusion, microstructure module, reinforcement-style objectives.

## 10. Deliverables
- Modular codebase under `models/transformers/` with documented APIs.
- Configuration templates and reproducible training/backtesting scripts.
- Experiment reports (Markdown/notebooks) with quantitative benchmarks vs. baselines.
- Documentation of tokenizer grammar and usage guidelines.
- Optional container/environment definition for deployment or batch inference.
