short answer: yes—transformers can work on both raw OHLCV and “price-action tokens.” They’ve been used in finance (OHLCV, order-book, candlestick-image and multimodal setups). Whether they beat CNN/LSTM depends on your horizon, data volume, and label. There’s also a growing body of work on discretizing continuous time series into tokens (SAX, VQ-VAE/TOTEM/TimeVQVAE) and on “word2vec-style” embeddings for assets (Stock2Vec / asset-embeddings). Below is a compact research map + a practical path to do this with an Al-Brooks-style tokenizer.

what’s already been done

Transformers on raw OHLCV / multivariate TS
	•	Temporal Fusion Transformer (TFT) and follow-ups have been applied to equities/crypto; several papers report good short-term forecasting when enriched with technical/ calendar features.  ￼
	•	PatchTST (ICLR’23) treats a time series as “patch tokens,” typically outperforming older transformer variants on long-horizon TS and is now a common baseline.  ￼
	•	iTransformer (ICLR’24) “inverts” tokenization (tokens are series/variates, not timesteps) and hits strong SOTA on multivariate TS; it’s been used for financial forecasting as well.  ￼
	•	Many finance-specific implementations exist (e.g., “Stockformer,” OHLCFormer repos, etc.).  ￼

Transformers on limit-order-book (LOB) / microstructure
	•	TransLOB combines causal CNN feature extraction with masked self-attention for mid-price moves. Subsequent variants (TLOB) add dual attention and report SOTA on FI-2010 and single-name LOB.  ￼

Candlestick charts as images (ViT/CNN)
	•	Vision Transformers and CNNs have been trained on candlestick images (often via GAF/spectrogram encodings) to classify patterns or predict direction.  ￼

Multimodal (prices + text/news/sentiment)
	•	MM-iTransformer fuses textual embeddings with price history via attention; similar “CandleFusion” demos mix candlestick visuals with text.  ￼

Discretizing continuous TS into tokens
	•	SAX turns series into symbolic strings (classic, often used in finance).  ￼
	•	VQ-VAE tokenizers (TOTEM, TimeVQVAE) learn a discrete codebook so you can train an autoregressive Transformer over tokens. Recent finance-focused writeups show promise.  ￼

“word2vec-style” finance embeddings
	•	Stock2Vec (and related “asset embeddings”) learn dense vectors for stocks from returns/holdings; these augment downstream models.  ￼

how this maps to “price-action tokens” (Al Brooks)

There isn’t a peer-reviewed, off-the-shelf “Brooks-token Transformer,” but the building blocks exist. Brooks’ public material lists the core primitives—trends vs ranges, wedges, channels, breakouts, measured moves, magnets/support, major trend reversals, L1/L2 pullbacks, etc.—which can be programmatically detected and serialized as tokens.  ￼

A practical tokenizer spec you can implement:
	•	Bar-level token: bull/bear/doji + body/upper-tail/lower-tail quantiles (e.g., 5-bin discretization), relative close vs range, relative volume.
	•	Swing structure token: HH/HL/LH/LL; pullback depth terciles; “micro-channel length.”
	•	Context token: regime (trend/range by volatility-adjusted directional movement), proximity to magnets (yesterday’s H/L, moving average, prior range midpoint), gap/open type.
	•	Event token: breakout attempt (with strength), failed breakout, measured-move target hit/missed.
You can learn embeddings for these tokens (like words) and model sequences with a Transformer; or learn the tokens themselves via VQ-VAE over normalized candle patches, then train an autoregressive prior. (SAX and VQ-VAE/TOTEM give you two proven discretization routes.)  ￼

“raw OHLCV vs tokens”—what to feed the model?
	•	Raw, continuous is absolutely fine. Transformers don’t require discrete tokens: apply a linear projection (or 1-D conv) to each timestep’s feature vector to get a learned embedding; add positional/time embeddings (Time2Vec or Fourier features) and go. PatchTST’s patching + instance-norm trick is a strong template.  ￼
	•	Discrete tokens help if you want interpretability, compression, or pretraining with language-model tricks (masking/next-token). SAX (fast, rule-based) or VQ-VAE (learned codebook) are the two common choices.  ￼

Should we “embed the financial data already continuous before feeding”?
Yes—use a learned projection layer to map each continuous feature vector (e.g., [log-return, hl-range, oc-position, vol…]) into the model’s d_model space, plus explicit time encodings (intraday minute-of-day, day-of-week) using Time2Vec/Fourier features. This is the standard recipe.  ￼

are CNN/LSTM more suitable than transformers?

Evidence is mixed, task-dependent:
	•	LOB / very short horizons: CNN or CNN-LSTM often remain competitive or superior; early “TransLOB” showed gains over LSTM in some setups, but subsequent studies note that simpler MLP/CNN baselines can still win depending on label design and costs.  ￼
	•	Longer horizons / many exogenous features: Transformers (TFT, PatchTST, iTransformer) frequently outperform RNN/CNN baselines by capturing long-range dependencies and cross-series structure.  ￼
	•	Caveat: Large surveys show linear baselines sometimes rival or beat complex Transformers on generic TS; finance is noisy and non-stationary, so regularization and evaluation on trading metrics (not just RMSE) matter.  ￼
	•	Direct head-to-heads in finance report mixed results (some favor Transformer, others find LSTM more stable).  ￼

concrete build recipes (what I’d actually try)
	1.	Raw-series Transformer (strong baseline)

	•	Features: log-returns, HL range %, close-in-range, volume (log), rolling z-scores; calendar Time2Vec; optional macro/news embeddings.
	•	Model: PatchTST or iTransformer; instance normalization; multi-horizon head; uncertainty via quantile loss.  ￼
	•	Multimodal add-on: fuse FinBERT/LLM text embeddings with price via cross-attention (as in MM-iTransformer).  ￼

	2.	Brooks-style “PriceActionTokenizer” + Transformer

	•	Detect Brooks primitives → emit tokens (see spec above).
	•	Train token-LM (masked modeling / next-token) for pretraining; fine-tune for classification (up/down/flat) or policy outputs.
	•	Alternative: learn tokens with VQ-VAE over normalized OHLCV patches (TOTEM/TimeVQVAE), then autoregress over codebook indices.  ￼

	3.	Microstructure track (if you have LOB)

	•	Start with CNN-LSTM or TransLOB; if spreading across instruments/horizons, try TLOB dual-attention. Evaluate in trading space (P&L after spread/slippage), not just F1.  ￼

notes on labels & evaluation (crucial in finance)
	•	Prefer classification on move bins (after spread) or risk-adjusted targets (Sharpe, hit-ratio/MI) over raw MSE. Recent work optimizes Sharpe directly with TFT variants.  ￼
	•	Pretrain with self-supervised TS methods (TimeMAE / contrastive TS2Vec-style) across many assets, then fine-tune on your market.  ￼

tl;dr answers to your questions
	•	“Can transformers be applied to price-action tokens or raw OHLCV?”
Yes—to both. Raw OHLCV is standard with PatchTST/iTransformer/TFT; discretized “price-action tokens” are feasible via SAX or learned VQ-VAE tokens, and you can design a Brooks-style grammar to emit tokens.  ￼
	•	“Has it been done?”
Transformers on OHLCV/LOB/candlestick images and multimodal (news+price) are widely published; an explicit “Al-Brooks tokenizer” paper doesn’t appear in the literature, but closely related tokenization approaches exist.  ￼
	•	“Are CNN/LSTM more suitable?”
Sometimes—especially for local microstructure and short horizons or when data is limited. Transformers shine with longer context, many series/features, and with good pretraining/regularization. Test both.  ￼
	•	“Embed continuous data before the model?”
Yes: project each timestep’s continuous features into an embedding (linear/conv), add Time2Vec/Fourier time encodings. If you want LM-style training, discretize first (SAX or VQ-VAE).  ￼
	•	“Use something like word2vec / vec2vec?”
Yes: Stock2Vec/asset embeddings learn useful vectors for assets; Series-level self-supervised “series2vec” methods also exist. You can also train Word2Vec-style embeddings over your own price-action token sequences.  ￼

If you want, I can sketch a minimal PyTorch pipeline that (a) emits Brooks-style tokens from your Python adaptation, (b) trains a small VQ-VAE tokenizer, and (c) fine-tunes a PatchTST/iTransformer head for your target horizon—all with sensible defaults for scaling/normalization and trading-metric evaluation.
