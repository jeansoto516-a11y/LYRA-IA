from pathlib import Path

from core.tokenizer.bpe import BPETokenizer
from training.dataset import (
    deduplicate_paragraphs,
    encode_documents,
    normalize_text,
    save_tokens,
)

PROCESSED_DIR = Path("data/processed")
TOKENIZER_FILE = Path("data/tokenizer/lyra_tokenizer_v0.json")
OUTPUT_FILE = Path("data/processed/tokens.npy")


def main() -> None:
    files = sorted(PROCESSED_DIR.glob("*_clean.txt"))
    if not files:
        raise SystemExit(f"Nenhum arquivo *_clean.txt encontrado em {PROCESSED_DIR}")

    tokenizer = BPETokenizer.load(TOKENIZER_FILE)

    texts = []
    total_chars = 0
    for f in files:
        raw = f.read_text(encoding="utf-8")
        text = deduplicate_paragraphs(normalize_text(raw))
        texts.append(text)
        total_chars += len(text)
        print(f"{f.name}: {len(raw)} -> {len(text)} caracteres")

    ids = encode_documents(texts, tokenizer)

    meta = {
        "files": [f.name for f in files],
        "tokenizer": str(TOKENIZER_FILE),
        "vocab_size": tokenizer.vocab_size,
        "num_tokens": int(len(ids)),
        "num_chars": total_chars,
    }
    save_tokens(ids, OUTPUT_FILE, meta)

    print(f"\nTokens: {len(ids)}")
    print(f"Caracteres por token: {total_chars / len(ids):.2f}")
    print(f"Salvo em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()