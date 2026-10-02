from dataclasses import dataclass, asdict


@dataclass
class LyraConfig:
    vocab_size: int = 1500
    context_length: int = 128
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)