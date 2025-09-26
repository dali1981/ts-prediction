"""PyTorch datasets and loaders for sliding window forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:  # pragma: no cover
    pl = None

from ..tokenizers import BrooksTokenVocabulary
from .config import DataConfig


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        continuous: np.ndarray,
        targets: np.ndarray,
        tokens: Optional[np.ndarray] = None,
        future: Optional[np.ndarray] = None,
    ) -> None:
        self.continuous = torch.tensor(continuous, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.tokens = torch.tensor(tokens, dtype=torch.long) if tokens is not None else None
        self.future = torch.tensor(future, dtype=torch.float32) if future is not None else None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.continuous)

    def __getitem__(self, idx: int):
        features = {"continuous": self.continuous[idx]}
        if self.tokens is not None:
            features["tokens"] = self.tokens[idx]
        if self.future is not None:
            features["future"] = self.future[idx]
        return features, self.targets[idx]


@dataclass(slots=True)
class WindowGenerator:
    config: DataConfig
    vocab: Optional[BrooksTokenVocabulary] = None

    def __post_init__(self) -> None:
        if self.config.token_column and self.config.vocab_path:
            self.vocab = BrooksTokenVocabulary.from_json(Path(self.config.vocab_path))

    def _encode_tokens(self, series: pd.Series) -> np.ndarray:
        if self.vocab is None:
            return series.to_numpy(dtype=np.int64)
        return series.astype(str).map(self.vocab.encode).to_numpy(dtype=np.int64)

    def generate(
        self, frame: pd.DataFrame, include_future: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        features = self.config.features
        target_column = self.config.target
        if target_column not in frame.columns:
            raise ValueError(f"Target column {target_column} missing from frame")
        missing = set(features) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")

        tokens_array = None
        if self.config.token_column:
            if self.config.token_column not in frame.columns:
                raise ValueError(f"Token column {self.config.token_column} missing from frame")
            tokens_array = self._encode_tokens(frame[self.config.token_column])

        values = frame[features + [target_column]].dropna().to_numpy(dtype=np.float32)
        if tokens_array is not None:
            tokens_array = tokens_array[-len(values):]
        future_values = frame[features].dropna().to_numpy(dtype=np.float32) if include_future else None

        lookback = self.config.lookback
        horizon = self.config.horizon
        inputs, targets, token_windows, future_windows = [], [], [], []
        for idx in range(lookback, len(values) - horizon + 1):
            window = values[idx - lookback : idx, :-1]
            future = values[idx : idx + horizon, -1]
            inputs.append(window)
            targets.append(future)
            if tokens_array is not None:
                token_windows.append(tokens_array[idx - lookback : idx])
            if include_future and future_values is not None:
                future_windows.append(future_values[idx : idx + horizon])
        if not inputs:
            raise ValueError("Insufficient data to build sliding windows")
        continuous = np.stack(inputs)
        targets_arr = np.stack(targets)
        token_arr = np.stack(token_windows) if token_windows else None
        future_arr = np.stack(future_windows) if future_windows else None
        return continuous, targets_arr, token_arr, future_arr


@dataclass(slots=True)
class DataModuleBuilder:
    config: DataConfig

    def build_loader(self, frame: pd.DataFrame) -> DataLoader:
        generator = WindowGenerator(self.config)
        cont, targets, tokens, future = generator.generate(frame, include_future=self.config.include_future_features)
        dataset = SlidingWindowDataset(cont, targets, tokens, future)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

    def build_datamodule(
        self,
        frame: pd.DataFrame,
        val_fraction: Optional[float] = None,
        test_fraction: Optional[float] = None,
    ):
        if pl is None:  # pragma: no cover
            raise RuntimeError("pytorch_lightning is required for build_datamodule")

        generator = WindowGenerator(self.config)
        cont, targets, tokens, future = generator.generate(frame, include_future=self.config.include_future_features)
        total = len(cont)

        val_fraction = self.config.val_fraction if val_fraction is None else val_fraction
        test_fraction = self.config.test_fraction if test_fraction is None else test_fraction

        val_size = int(total * val_fraction)
        test_size = int(total * test_fraction)
        train_size = total - val_size - test_size
        if train_size <= 0:
            raise ValueError("Insufficient data after applying validation/test splits")

        def split(array, has_tokens: bool = False):
            if array is None:
                return None, None, None
            train = array[:train_size]
            val = array[train_size : train_size + val_size] if val_size else None
            test = array[train_size + val_size :] if test_size else None
            if has_tokens:
                return (train, val if val is not None else None, test if test is not None else None)
            return train, val, test

        cont_train, cont_val, cont_test = split(cont)
        tgt_train, tgt_val, tgt_test = split(targets)
        tok_train, tok_val, tok_test = split(tokens, has_tokens=True)
        fut_train, fut_val, fut_test = split(future)

        def _empty_like(arr: np.ndarray) -> np.ndarray:
            shape = list(arr.shape)
            shape[0] = 0
            return np.empty(shape, dtype=arr.dtype)

        def _empty_tokens(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if arr is None:
                return None
            shape = list(arr.shape)
            shape[0] = 0
            return np.empty(shape, dtype=arr.dtype)

        if cont_val is None:
            cont_val = _empty_like(cont_train)
            tgt_val = _empty_like(tgt_train)
            tok_val = _empty_tokens(tok_train)
            fut_val = _empty_tokens(fut_train)
        if cont_test is None:
            cont_test = _empty_like(cont_train)
            tgt_test = _empty_like(tgt_train)
            tok_test = _empty_tokens(tok_train)
            fut_test = _empty_tokens(fut_train)

        class SlidingWindowDataModule(pl.LightningDataModule):
            def __init__(self, batch_size: int) -> None:
                super().__init__()
                self.batch_size = batch_size
                self._train_data = SlidingWindowDataset(cont_train, tgt_train, tok_train, fut_train)
                self._val_data = SlidingWindowDataset(cont_val, tgt_val, tok_val, fut_val)
                self._test_data = SlidingWindowDataset(cont_test, tgt_test, tok_test, fut_test)

            def setup(self, stage: str | None = None) -> None:
                self.train_dataset = self._train_data
                self.val_dataset = self._val_data
                self.test_dataset = self._test_data

            def train_dataloader(self):
                return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

            def val_dataloader(self):  # pragma: no cover
                return DataLoader(self.val_dataset, batch_size=self.batch_size)

            def test_dataloader(self):  # pragma: no cover
                return DataLoader(self.test_dataset, batch_size=self.batch_size)

        return SlidingWindowDataModule(self.config.batch_size)
