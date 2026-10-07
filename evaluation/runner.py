"""Executa a avaliação completa de um checkpoint e compara resultados entre versões."""
import torch

from conversation.session import ChatSession
from evaluation import metrics as M

COPY_N = 6

# True = menor é melhor, False = maior é melhor, None = só informativo
DIRECTIONS = {
    "val.loss": True,
    "val.perplexity": True,
    "val.bits_per_byte": True,
    "test.loss": True,
    "test.perplexity": True,
    "test.bits_per_byte": True,
    "generation.distinct_1": False,
    "generation.distinct_2": False,
    "generation.invented_word_rate": True,
    "generation.replacement_char_rate": True,
    f"generation.copy_rate_{COPY_N}": None,
    "chat.turn_marker_leaks": True,
    "chat.empty_reply_rate": True,
    "chat.prompt_within_budget": False,
}


def evaluate_generation(engine, prompts, corpus_texts, settings) -> dict:
    texts = []
    for i, prompt in enumerate(prompts):
        torch.manual_seed(settings["seed"] + i)
        texts.append(
            engine.generate_text(
                prompt,
                max_new_tokens=settings["max_new_tokens"],
                temperature=settings["temperature"],
                top_k=settings["top_k"],
                top_p=settings["top_p"],
            )
        )
    vocabulary = {w for t in corpus_texts for w in M.words(t)}
    reference = M.corpus_ngrams(corpus_texts, COPY_N)
    return {
        "num_prompts": len(prompts),
        "distinct_1": M.distinct_n(texts, 1),
        "distinct_2": M.distinct_n(texts, 2),
        "invented_word_rate": M.invented_word_rate(texts, vocabulary),
        f"copy_rate_{COPY_N}": M.copy_rate(texts, reference, COPY_N),
        "replacement_char_rate": M.replacement_char_rate(texts),
    }


def evaluate_chat(engine, persona, script, settings) -> dict:
    session = ChatSession(engine, persona)
    leaks = 0
    empty = 0
    within_budget = True
    markers = (f"\n{persona.user_label}:", f"\n{persona.name}:")
    for i, user_text in enumerate(script):
        torch.manual_seed(settings["seed"] + 1000 + i)
        reply = session.send(
            user_text,
            temperature=settings["temperature"],
            top_k=settings["top_k"],
            top_p=settings["top_p"],
        )
        leaks += sum(1 for m in markers if m in reply)
        empty += 1 if not reply.strip() else 0
        if len(session.last_context.token_ids) > session.context.budget:
            within_budget = False
    return {
        "turns": len(script),
        "turn_marker_leaks": leaks,
        "empty_reply_rate": empty / len(script) if script else 0.0,
        "prompt_within_budget": within_budget,
    }


def run_evaluation(
    engine, dataset, corpus_texts, prompts_config, persona, name, step=None, settings=None
) -> dict:
    settings = settings or {
        "seed": 1234,
        "max_new_tokens": 60,
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.95,
    }
    ctx = engine.model.config.context_length
    return {
        "name": name,
        "checkpoint_step": step,
        "parameters": engine.model.num_parameters(),
        "vocab_size": engine.tokenizer.vocab_size,
        "context_length": ctx,
        "settings": settings,
        "val": M.evaluate_split(engine.model, dataset.splits["val"], engine.tokenizer, ctx),
        "test": M.evaluate_split(engine.model, dataset.splits["test"], engine.tokenizer, ctx),
        "generation": evaluate_generation(
            engine, prompts_config["completion_prompts"], corpus_texts, settings
        ),
        "chat": evaluate_chat(engine, persona, prompts_config["chat_script"], settings),
    }


def _flatten(result: dict) -> dict:
    flat = {}
    for section in ("val", "test", "generation", "chat"):
        for key, value in result[section].items():
            flat[f"{section}.{key}"] = value
    return flat


def compare(old: dict, new: dict) -> list[dict]:
    """Compara dois resultados e diz, métrica por métrica, se melhorou ou piorou."""
    a, b = _flatten(old), _flatten(new)
    rows = []
    for key, lower_is_better in DIRECTIONS.items():
        if key not in a or key not in b:
            continue
        va, vb = a[key], b[key]
        if lower_is_better is None or isinstance(va, bool):
            if isinstance(va, bool):
                status = "same" if va == vb else ("improved" if vb else "worse")
            else:
                status = "info"
        elif vb == va:
            status = "same"
        else:
            better = vb < va if lower_is_better else vb > va
            status = "improved" if better else "worse"
        rows.append({"metric": key, "old": va, "new": vb, "status": status})
    return rows