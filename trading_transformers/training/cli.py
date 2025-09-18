"""Command-line interface for running transformer experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for the training CLI") from exc

from ..data import DataCatalog
from .config import ExperimentConfig
from .runner import ExperimentRunner


def run_experiment(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    with config_path.open() as handle:
        payload = yaml.safe_load(handle)
    experiment = ExperimentConfig.from_dict(payload)

    if args.accelerator:
        experiment.trainer.accelerator = args.accelerator
    if args.devices:
        experiment.trainer.devices = args.devices if args.devices.isdigit() is False else int(args.devices)

    catalog_path = Path(args.catalog)
    catalog = DataCatalog.from_json(catalog_path)

    runner = ExperimentRunner(config=experiment, catalog=catalog)
    trainer = runner.run()
    log_dir = getattr(trainer, "log_dir", None)
    if log_dir is None:
        log_dir = experiment.output_dir
    print(f"Training completed. Checkpoints stored in {log_dir}")
    if runner.report:
        print("Diagnostics:")
        for key, value in runner.report.items():
            print(f"  {key}: {value}")
    if args.diagnostics and runner.report:
        out_path = Path(args.diagnostics)
        out_path.write_text(json.dumps(runner.report, indent=2))
        print(f"Diagnostics saved to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transformers-train", description="Run transformer experiments")
    parser.add_argument("--config", required=True, help="Experiment YAML path")
    parser.add_argument("--catalog", required=True, help="Catalog JSON path")
    parser.add_argument("--diagnostics", help="Optional path to write diagnostics JSON")
    parser.add_argument("--accelerator", help="Override accelerator (cpu, mps, gpu, etc.)")
    parser.add_argument("--devices", help="Override devices specification (e.g., 1, auto)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
