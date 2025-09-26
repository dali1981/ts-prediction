# Data Plan

This plan coordinates catalog hygiene, feature pipelines, and model hand-offs so that both transformer experiments and alternative baselines share the same data substrates.

## Objectives

1. **Stabilise ingestion** — make catalog paths portable and guard against token/feature misalignment.
2. **Unify feature schemas** — define reusable transformations for continuous and token features with explicit versioning.
3. **Support mixed model families** — ensure loaders emit shapes and metadata compatible with transformers, Chronos, LSTM, CNN, and future entrants.
4. **Automate validation** — add coverage that detects schema drift, NaN handling issues, and catalog regressions early in CI.

## Workstreams & Milestones

### A. Catalog & Storage (Weeks 1-2)
- Normalise relative paths against `catalog.root` when loading JSON catalogs.
- Provide checksum verification for optional Parquet/CSV sources (mirror the existing zip verification).
- Add CLI flag to rebase catalogs when data directories move.

### B. Feature Engineering (Weeks 2-4)
- Freeze a baseline feature spec (`continuous_v1`) with documented column names and units.
- Implement a deterministic token encoder that maps string labels → integer ids without external vocab (fallback mode).
- Document how to plug custom feature calculators (macro data, order book signals) into the staging loop.

### C. Windowing & Loaders (Weeks 3-5)
- Fix token alignment by applying the same NaN mask to both continuous features and token arrays.
- Introduce validation hooks that confirm window counts and sequence shapes for each model type.
- Export a thin adapter that converts window outputs to the Chronos dataset format for quick benchmarking.

### D. Quality & Tooling (Weeks 4-6)
- Write property-based tests for `WindowGenerator` covering NaN, short series, and mixed-frequency inputs.
- Add smoke tests that load the catalog, build windows, and run a minimal forward pass for each supported model family.
- Integrate data profiling (e.g., pandas-profiling or Great Expectations) on nightly builds to surface schema drift.

## Deliverables

- `docs/findings_20250926.md` — live issue log (complete).
- `docs/data_pipeline.md` — updated loader documentation (complete).
- Automated test suite updates (pending in follow-up PR).
- CI checks gating catalog and window regressions (pending).

## Risks & Mitigations

- **Schema drift from external sources:** version the feature config and include checksum snapshots.
- **Token logic blocking non-transformer models:** keep token handling optional and provide pure-continuous adapters for Chronos/LSTM/CNN flows.
- **Catalog portability delays:** prioritise path rebasing before onboarding additional datasets.

Tracking these milestones will keep the data layer healthy while new model experiments come online.
