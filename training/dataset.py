"""Pipeline de dados: limpeza, tokenização, divisão train/val/test e batches."""
import json
import unicodedata
from pathlib import Path

import numpy as np
import torch

MIN_DEDUP_LEN = 80  # só parágrafos longos são considerados para deduplicação


def normalize_text(text: str) -> str:
    """Normaliza acentos (NFC) e quebras de linha."""
    text = unicodedata.normalize("NFC", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def deduplicate_paragraphs(text: str, min_len: int = MIN_DEDUP_LEN) -> str:
    """Remove parágrafos longos repetidos (mantém a primeira ocorrência)."""
    seen = set()
    kept = []
    for paragraph in text.split("\n\n"):
        key = paragraph.strip()
        if len(key) >= min_len:
            if key in seen:
                continue
            seen.add(key)
        kept.append(paragraph)
    return "\n\n".join(kept)


def encode_documents(texts: list[str], tokenizer) -> np.ndarray:
    """Tokeniza cada documento entre <bos> e <eos> e junta tudo em uma sequência."""
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    return np.asarray(ids, dtype=np.int32)


def save_tokens(ids: np.ndarray, path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, ids)
    path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_tokens(path) -> np.ndarray:
    return np.load(Path(path))


class TokenDataset:
    """Divide a sequência de tokens em train/val/test (em ordem) e gera batches."""

    def __init__(self, ids, val_fraction: float = 0.05, test_fraction: float = 0.05):
        if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
            raise ValueError("Frações de val/test inválidas")
        data = torch.as_tensor(np.asarray(ids), dtype=torch.long)
        n = len(data)
        n_test = int(n * test_fraction)
        n_val = int(n * val_fraction)
        n_train = n - n_val - n_test
        self.splits = {
            "train": data[:n_train],
            "val": data[n_train : n_train + n_val],
            "test": data[n_train + n_val :],
        }

    def sizes(self) -> dict:
        return {name: len(t) for name, t in self.splits.items()}

    def get_batch(self, split: str, batch_size: int, block_size: int, generator=None):
        data = self.splits[split]
        if len(data) <= block_size + 1:
            raise ValueError(
                f"Split '{split}' tem só {len(data)} tokens; precisa de mais que "
                f"{block_size + 1}"
            )
        starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
        x = torch.stack([data[i : i + block_size] for i in starts])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
        return x, y