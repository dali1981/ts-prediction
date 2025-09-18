"""Command-line helpers for data catalog management."""

from __future__ import annotations

import argparse
from pathlib import Path

from .catalog import DataCatalog
from .sources import auto_register_archives, register_csv_folder


def init_catalog(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    catalog = DataCatalog(root=root)
    if args.zip_dir:
        auto_register_archives(catalog, Path(args.zip_dir))
    if args.csv_dir:
        register_csv_folder(catalog, Path(args.csv_dir), frequency=args.frequency)
    catalog_path = Path(args.output or (root / "catalog.json"))
    catalog.to_json(catalog_path)
    print(f"Catalog written to {catalog_path}")


def list_entries(args: argparse.Namespace) -> None:
    catalog = DataCatalog.from_json(Path(args.catalog))
    print("Sources:")
    for name in catalog.list_sources():
        print(f"  - {name}")
    print("Archives:")
    for name in catalog.list_archives():
        print(f"  - {name}")


def extract_archive(args: argparse.Namespace) -> None:
    catalog = DataCatalog.from_json(Path(args.catalog))
    target = catalog.extract_archive(args.name)
    print(f"Extracted to {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transformers-data", description="Data catalog utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init", help="Create a catalog file")
    init_cmd.add_argument("--root", required=True, help="Catalog root directory")
    init_cmd.add_argument("--zip-dir", help="Directory containing zip archives")
    init_cmd.add_argument("--csv-dir", help="Directory containing CSV files")
    init_cmd.add_argument("--frequency", help="Frequency label for registered CSV files")
    init_cmd.add_argument("--output", help="Catalog JSON output path")
    init_cmd.set_defaults(func=init_catalog)

    list_cmd = sub.add_parser("list", help="List registered sources and archives")
    list_cmd.add_argument("--catalog", required=True, help="Path to catalog JSON")
    list_cmd.set_defaults(func=list_entries)

    extract_cmd = sub.add_parser("extract", help="Extract a registered archive")
    extract_cmd.add_argument("--catalog", required=True, help="Path to catalog JSON")
    extract_cmd.add_argument("name", help="Archive name")
    extract_cmd.set_defaults(func=extract_archive)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
