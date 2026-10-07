"""Métricas de avaliação da Lyra: modelagem de linguagem e qualidade do texto gerado."""
import math
import re

import torch
import torch.nn.functional as F

WORD_PATTERN = re.compile(r"[^\W\d_]+")


def words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


def token_byte_lengths(tokenizer) -> torch.Tensor:
    """Quantos bytes cada token representa (tokens especiais valem 0)."""
    return torch.tensor(
        [len(tokenizer.vocab.get(i, b"")) for i in range(tokenizer.vocab_size)],
        dtype=torch.long,
    )


@torch.no_grad()
def evaluate_split(model, ids, tokenizer, context_length: int, batch_size: int = 16) -> dict:
    """Loss, perplexidade e bits por byte sobre TODA a sequência.

    Usa janelas sem sobreposição de tamanho context_length. Como o resultado não
    depende de sorteio, é exatamente reproduzível. Bits por byte não depende do
    tamanho do vocabulário, então serve para comparar tokenizers diferentes.
    """
    data = torch.as_tensor(ids, dtype=torch.long)
    starts = list(range(0, len(data) - context_length, context_length))
    if not starts:
        raise ValueError("Sequência curta demais para avaliar")

    byte_len = token_byte_lengths(tokenizer)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    for b in range(0, len(starts), batch_size):
        chunk = starts[b : b + batch_size]
        x = torch.stack([data[s : s + context_length] for s in chunk]).to(device)
        y = torch.stack([data[s + 1 : s + context_length + 1] for s in chunk]).to(device)
        logits, _ = model(x)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        )
        total_nll += nll.item()
        total_tokens += y.numel()
        total_bytes += int(byte_len[y.cpu()].sum())
    model.train(was_training)

    loss = total_nll / total_tokens
    return {
        "tokens": total_tokens,
        "loss": loss,
        "perplexity": math.exp(loss),
        "bits_per_byte": total_nll / math.log(2) / max(1, total_bytes),
    }


def distinct_n(texts: list[str], n: int) -> float:
    """Fração de n-gramas de palavras que são únicos (baixo = muita repetição)."""
    grams = []
    for t in texts:
        w = words(t)
        grams.extend(tuple(w[i : i + n]) for i in range(len(w) - n + 1))
    return len(set(grams)) / len(grams) if grams else 0.0


def invented_word_rate(texts: list[str], vocabulary: set[str]) -> float:
    """Fração das palavras geradas que não existem no corpus de treino."""
    gen = [w for t in texts for w in words(t)]
    if not gen:
        return 0.0
    return sum(1 for w in gen if w not in vocabulary) / len(gen)


def corpus_ngrams(corpus_texts: list[str], n: int) -> set[tuple]:
    grams: set[tuple] = set()
    for text in corpus_texts:
        w = words(text)
        grams.update(tuple(w[i : i + n]) for i in range(len(w) - n + 1))
    return grams


def copy_rate(texts: list[str], reference_ngrams: set[tuple], n: int) -> float:
    """Fração dos n-gramas gerados que aparecem exatamente no corpus (memorização)."""
    gen = []
    for t in texts:
        w = words(t)
        gen.extend(tuple(w[i : i + n]) for i in range(len(w) - n + 1))
    if not gen:
        return 0.0
    return sum(1 for g in gen if g in reference_ngrams) / len(gen)


def replacement_char_rate(texts: list[str]) -> float:
    """Fração dos textos com bytes inválidos (aparecem como �)."""
    if not texts:
        return 0.0
    return sum(1 for t in texts if "\ufffd" in t) / len(texts)