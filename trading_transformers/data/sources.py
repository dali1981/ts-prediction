"""Helpers to build data catalogs from repository structure."""

from __future__ import annotations

from pathlib import Path
from .catalog import DataCatalog, DataSource, ZipArchive


def auto_register_archives(catalog: DataCatalog, search_path: Path) -> None:
    for zip_path in search_path.glob("*.zip"):
        name = zip_path.stem
        target = catalog.cache_dir / name
        catalog.register_zip(ZipArchive(name=name, path=zip_path, target_dir=target))


def register_csv_folder(catalog: DataCatalog, folder: Path, frequency: str | None = None) -> None:
    for csv_file in folder.glob("*.csv"):
        name = csv_file.stem
        catalog.register_source(
            DataSource(name=name, path=csv_file, fmt="csv", frequency=frequency)
        )
