import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.model.config import LyraConfig


@dataclass
class TrainConfig:
    model: LyraConfig = field(default_factory=LyraConfig)
    seed: int = 1337

    tokens_path: str = "data/processed/tokens.npy"
    checkpoint_dir: str = "checkpoints/lyra_0_1"
    val_fraction: float = 0.05
    test_fraction: float = 0.05

    batch_size: int = 16
    grad_accum_steps: int = 2
    max_steps: int = 2000
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 25
    patience: int = 5
    min_delta: float = 0.0

    device: str = "auto"
    mixed_precision: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        data = dict(data)
        model = LyraConfig(**data.pop("model", {}))
        return cls(model=model, **data)

    @classmethod
    def from_json(cls, path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)