import os
from pathlib import Path

import torch


def save_checkpoint(path, state: dict) -> None:
    """Salva de forma atômica: escreve num arquivo temporário e depois troca."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, map_location="cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=True)