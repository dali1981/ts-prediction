"""Simple backtesting CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to use the backtest CLI") from exc

from .evaluation.backtest import Backtester


def load_config(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def run_backtest(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    data_cfg = config.get("data", {})
    csv_path = Path(data_cfg["csv"])
    timestamp_col = data_cfg.get("timestamp_column")
    parse_dates = [timestamp_col] if data_cfg.get("parse_dates") and timestamp_col else None
    df = pd.read_csv(csv_path, parse_dates=parse_dates)

    signal_col = data_cfg["signal_column"]
    returns_col = data_cfg["returns_column"]
    signals = df[signal_col]
    returns = df[returns_col]

    backtest_cfg = config.get("backtest", {})
    transaction_cost = backtest_cfg.get("transaction_cost", 0.0)

    backtester = Backtester(transaction_cost=transaction_cost)
    metrics = backtester.run(signals=signals, returns=returns)

    print("Backtest results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transformers-backtest", description="Run backtests on prediction files")
    parser.add_argument("--config", required=True, help="Backtest YAML configuration")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_backtest(args)


if __name__ == "__main__":  # pragma: no cover
    main()
