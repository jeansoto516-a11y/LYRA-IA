"""Tokenizer BPE (Byte Pair Encoding) em nível de bytes, escrito do zero.

Ideia: todo texto vira bytes UTF-8, então nunca existe caractere "desconhecido".
O treino descobre os pares de símbolos mais frequentes e os funde em tokens novos.
"""
import json
import re
from collections import Counter
from pathlib import Path

# Separa o texto em "palavras" (com o espaço colado na frente), números e pontuação.
# [^\W\d_] casa letras Unicode, incluindo acentos (ç, ã, é...).
PRETOKEN_PATTERN = re.compile(r"\s?[^\W\d_]+|\s?\d+|\s?[^\s\w]+|\s+(?!\S)|\s+")

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]
NUM_SPECIAL = len(SPECIAL_TOKENS)
BYTE_OFFSET = NUM_SPECIAL  # o byte b vira o id (b + BYTE_OFFSET)
BASE_VOCAB_SIZE = NUM_SPECIAL + 256


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []
        self._rebuild()

    # ------------------------------------------------------------------ interno
    def _rebuild(self) -> None:
        """Reconstrói as tabelas auxiliares a partir da lista de merges."""
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {
            b + BYTE_OFFSET: bytes([b]) for b in range(256)
        }
        for i, (a, b) in enumerate(self.merges):
            self.vocab[BASE_VOCAB_SIZE + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return BASE_VOCAB_SIZE + len(self.merges)

    @property
    def pad_id(self) -> int:
        return SPECIAL_TOKENS.index("<pad>")

    @property
    def bos_id(self) -> int:
        return SPECIAL_TOKENS.index("<bos>")

    @property
    def eos_id(self) -> int:
        return SPECIAL_TOKENS.index("<eos>")

    @staticmethod
    def _pretokenize(text: str) -> list[str]:
        return PRETOKEN_PATTERN.findall(text)

    # ------------------------------------------------------------------- treino
    def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(f"vocab_size precisa ser >= {BASE_VOCAB_SIZE}")

        word_freq = Counter(self._pretokenize(text))
        words = {
            tuple(b + BYTE_OFFSET for b in w.encode("utf-8")): f
            for w, f in word_freq.items()
        }

        self.merges = []
        num_merges = vocab_size - BASE_VOCAB_SIZE

        for step in range(num_merges):
            pair_counts: Counter = Counter()
            for symbols, freq in words.items():
                for pair in zip(symbols, symbols[1:]):
                    pair_counts[pair] += freq
            if not pair_counts:
                break

            best_pair, best_count = max(
                pair_counts.items(), key=lambda kv: (kv[1], kv[0])
            )
            if best_count < 2:
                break

            new_id = BASE_VOCAB_SIZE + len(self.merges)
            self.merges.append(best_pair)

            new_words = {}
            for symbols, freq in words.items():
                new_words[self._merge_symbols(symbols, best_pair, new_id)] = freq
            words = new_words

            if verbose and (step + 1) % 100 == 0:
                print(f"  merge {step + 1}/{num_merges} (par mais comum: {best_count}x)")

        self._rebuild()

    @staticmethod
    def _merge_symbols(symbols, pair, new_id):
        out = []
        i = 0
        n = len(symbols)
        while i < n:
            if i < n - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return tuple(out)

    # ------------------------------------------------------- encode / decode
    def _encode_word(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached

        ids = [b + BYTE_OFFSET for b in word.encode("utf-8")]
        while len(ids) >= 2:
            best_rank = None
            best_pos = -1
            for i in range(len(ids) - 1):
                rank = self.merge_ranks.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pos = i
            if best_rank is None:
                break
            ids[best_pos : best_pos + 2] = [BASE_VOCAB_SIZE + best_rank]

        self._cache[word] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for word in self._pretokenize(text):
            ids.extend(self._encode_word(word))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return data.decode("utf-8", errors="replace")

    # ----------------------------------------------------------- persistência
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "byte-level-bpe",
            "version": 1,
            "special_tokens": SPECIAL_TOKENS,
            "merges": [list(p) for p in self.merges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("special_tokens") != SPECIAL_TOKENS:
            raise ValueError("Tokens especiais incompatíveis com esta versão.")
        tok = cls()
        tok.merges = [tuple(p) for p in payload["merges"]]
        tok._rebuild()
        return tok