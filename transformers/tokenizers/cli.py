"""CLI for Brooks tokenizer analytics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from .analytics import compute_stats, cooccurrence_matrix


def _read_tokens(path: Path, column: str | None) -> Iterable[str]:
    if column:
        df = pd.read_csv(path)
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in {path}")
        return df[column].astype(str).tolist()
    return Path(path).read_text().split()


def run_stats(args: argparse.Namespace) -> None:
    tokens = _read_tokens(Path(args.input), args.column)
    stats = compute_stats(tokens, top_k=args.top_k)
    print("Token statistics:")
    print(f"  total_tokens: {stats.total_tokens}")
    print(f"  unique_tokens: {stats.unique_tokens}")
    print(f"  entropy: {stats.entropy:.4f}")
    print("  top_tokens:")
    for token, count in stats.top_tokens:
        print(f"    {token}: {count}")


def run_cooccurrence(args: argparse.Namespace) -> None:
    path = Path(args.input)
    df = pd.read_csv(path)
    if args.sequence_column not in df.columns:
        raise ValueError(f"Column {args.sequence_column} not found in {path}")
    sequences = df[args.sequence_column].apply(lambda x: str(x).split(args.delimiter)).tolist()
    matrix = cooccurrence_matrix(sequences)
    output = Path(args.output)
    matrix.to_csv(output)
    print(f"Co-occurrence matrix saved to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transformers-tokens", description="Brooks token analytics")
    sub = parser.add_subparsers(dest="command", required=True)

    stats_cmd = sub.add_parser("stats", help="Compute token statistics")
    stats_cmd.add_argument("--input", required=True, help="Path to token file (txt or CSV)")
    stats_cmd.add_argument("--column", help="CSV column containing tokens")
    stats_cmd.add_argument("--top-k", type=int, default=20, help="Number of top tokens to display")
    stats_cmd.set_defaults(func=run_stats)

    co_cmd = sub.add_parser("cooccurrence", help="Build co-occurrence matrix from token sequences")
    co_cmd.add_argument("--input", required=True, help="CSV file with token sequences")
    co_cmd.add_argument("--sequence-column", required=True, help="Column containing space-delimited token sequences")
    co_cmd.add_argument("--delimiter", default=" ", help="Delimiter between tokens in the sequence column")
    co_cmd.add_argument("--output", required=True, help="Output CSV path for co-occurrence matrix")
    co_cmd.set_defaults(func=run_cooccurrence)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
