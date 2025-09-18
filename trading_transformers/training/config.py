"""Experiment configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass(slots=True)
class DataConfig:
    source: str
    features: List[str]
    target: str
    lookback: int
    horizon: int
    batch_size: int = 64
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    token_column: Optional[str] = None
    vocab_path: Optional[str] = None


@dataclass(slots=True)
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4


@dataclass(slots=True)
class TrainerConfig:
    max_epochs: int = 50
    accelerator: str = "auto"
    precision: str = "16-mixed"
    gradient_clip_val: float = 1.0
    devices: Optional[Union[int, str]] = None


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    data: DataConfig
    model: Dict[str, str]
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    output_dir: Path = Path("artifacts")

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ExperimentConfig":
        data_cfg = DataConfig(**payload["data"])
        model_cfg = payload.get("model", {})
        optimizer_cfg = OptimizerConfig(**payload.get("optimizer", {}))
        trainer_cfg = TrainerConfig(**payload.get("trainer", {}))
        output_dir = Path(payload.get("output_dir", "artifacts"))
        return cls(
            name=payload.get("name", "experiment"),
            data=data_cfg,
            model=model_cfg,
            optimizer=optimizer_cfg,
            trainer=trainer_cfg,
            output_dir=output_dir,
        )

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "name": self.name,
            "data": self.data.__dict__,
            "model": self.model,
            "optimizer": self.optimizer.__dict__,
            "trainer": self.trainer.__dict__,
            "output_dir": str(self.output_dir),
        }
        return payload
