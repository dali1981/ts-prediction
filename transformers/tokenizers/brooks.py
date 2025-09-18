"""Utilities for handling Brooks-style token vocabularies."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import json


@dataclass(slots=True)
class BrooksTokenVocabulary:
    tokens: Dict[str, int]

    @classmethod
    def from_sequences(cls, sequences: Iterable[str]) -> "BrooksTokenVocabulary":
        counter = Counter(sequences)
        # Reserve 0 for padding, 1 for unknown
        vocab = {"<pad>": 0, "<unk>": 1}
        for idx, (token, _) in enumerate(counter.most_common(), start=2):
            vocab[token] = idx
        return cls(tokens=vocab)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.tokens, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "BrooksTokenVocabulary":
        return cls(tokens=json.loads(path.read_text()))

    def encode(self, token: str) -> int:
        return self.tokens.get(token, self.tokens["<unk>"])

    def decode(self, index: int) -> str:
        for token, idx in self.tokens.items():
            if idx == index:
                return token
        return "<unk>"
