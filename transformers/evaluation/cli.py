"""Evaluation CLI for computing predictive metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for the evaluation CLI") from exc

from .metrics import MetricsBundle


def load_config(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def run_metrics(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    data_cfg = config.get("data", {})
    csv_path = Path(data_cfg["csv"])
    df = pd.read_csv(csv_path)

    y_true = df[data_cfg["target_column"]]
    y_pred = df[data_cfg["prediction_column"]]

    bundle = MetricsBundle()
    metrics = {}
    if config.get("regression", True):
        metrics.update(bundle.regression(y_true, y_pred))
    if config.get("directional", True):
        metrics.update(bundle.directional(y_true, y_pred))

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transformers-eval", description="Compute metrics for prediction files")
    parser.add_argument("--config", required=True, help="Evaluation YAML configuration")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_metrics(args)


if __name__ == "__main__":  # pragma: no cover
    main()
