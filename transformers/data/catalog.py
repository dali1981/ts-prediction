"""Dataset catalog and ingestion helpers."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass(slots=True)
class DataSource:
    name: str
    path: Path
    fmt: str
    frequency: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "path": str(self.path),
            "fmt": self.fmt,
            "frequency": self.frequency or "",
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ZipArchive:
    name: str
    path: Path
    target_dir: Path
    checksum: Optional[str] = None

    def verify(self) -> None:
        if not self.checksum:
            return
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        if digest.hexdigest() != self.checksum:
            raise ValueError(f"Checksum mismatch for {self.path}")


class DataCatalog:
    """Registry for tabular financial datasets."""

    def __init__(self, root: str | Path, cache_dir: Optional[Path] = None) -> None:
        self.root = Path(root)
        self.cache_dir = Path(cache_dir) if cache_dir else self.root / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sources: Dict[str, DataSource] = {}
        self.archives: Dict[str, ZipArchive] = {}

    # Registration -----------------------------------------------------------------
    def register_source(self, source: DataSource) -> None:
        if source.name in self.sources:
            raise KeyError(f"Source {source.name} already registered")
        self.sources[source.name] = source

    def register_zip(self, archive: ZipArchive) -> None:
        if archive.name in self.archives:
            raise KeyError(f"Archive {archive.name} already registered")
        self.archives[archive.name] = archive

    # Loading ----------------------------------------------------------------------
    def load(self, name: str, **kwargs) -> pd.DataFrame:
        source = self.sources.get(name)
        if not source:
            raise KeyError(f"Unknown source {name}")
        path = source.path
        if source.fmt == "csv":
            return pd.read_csv(path, **kwargs)
        if source.fmt == "parquet":
            return pd.read_parquet(path, **kwargs)
        raise ValueError(f"Unsupported format {source.fmt}")

    def extract_archive(self, name: str) -> Path:
        archive = self.archives.get(name)
        if not archive:
            raise KeyError(f"Unknown archive {name}")
        archive.verify()
        target = archive.target_dir
        target.mkdir(parents=True, exist_ok=True)
        marker = target / ".extracted"
        if marker.exists():
            return target
        with zipfile.ZipFile(archive.path) as zf:
            zf.extractall(target)
        marker.write_text("ok")
        return target

    def load_from_archive(self, archive_name: str, member: str, **kwargs) -> pd.DataFrame:
        folder = self.extract_archive(archive_name)
        path = folder / member
        if not path.exists():
            raise FileNotFoundError(f"{member} not found in archive {archive_name}")
        if path.suffix == ".csv":
            return pd.read_csv(path, **kwargs)
        if path.suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path, **kwargs)
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Persistence ------------------------------------------------------------------
    def to_json(self, path: Path) -> None:
        data = {
            "root": str(self.root),
            "sources": [src.to_dict() for src in self.sources.values()],
            "archives": [
                {
                    "name": arc.name,
                    "path": str(arc.path),
                    "target_dir": str(arc.target_dir),
                    "checksum": arc.checksum or "",
                }
                for arc in self.archives.values()
            ],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "DataCatalog":
        blob = json.loads(path.read_text())
        catalog = cls(Path(blob["root"]))
        for src in blob.get("sources", []):
            catalog.register_source(
                DataSource(
                    name=src["name"],
                    path=Path(src["path"]),
                    fmt=src["fmt"],
                    frequency=src.get("frequency") or None,
                    metadata=src.get("metadata", {}),
                )
            )
        for arc in blob.get("archives", []):
            catalog.register_zip(
                ZipArchive(
                    name=arc["name"],
                    path=Path(arc["path"]),
                    target_dir=Path(arc["target_dir"]),
                    checksum=arc.get("checksum") or None,
                )
            )
        return catalog

    # Utility ----------------------------------------------------------------------
    def list_sources(self) -> List[str]:
        return sorted(self.sources)

    def list_archives(self) -> List[str]:
        return sorted(self.archives)
