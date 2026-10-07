import json
import math

import pytest
import torch
import torch.nn as nn

from core.inference.engine import LyraEngine
from core.model.config import LyraConfig
from core.model.lyra import LyraModel
from core.tokenizer.bpe import BPETokenizer
from evaluation import metrics as M
from evaluation.runner import compare, run_evaluation
from personality.persona import Persona
from training.dataset import TokenDataset

CORPUS = (
    "Lyra é uma assistente educada e objetiva. Usuário: olá, como vai? "
    "Lyra: vou bem, obrigada. Ela admite quando não sabe. "
) * 60


@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=400, verbose=False)
    return t


class ModeloUniforme(nn.Module):
    """Modelo de mentira que dá a mesma probabilidade para todos os tokens."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros(x.size(0), x.size(1), self.vocab) + self.p, None


# ---------------------------------------------------------- métricas de texto
def test_words_ignora_pontuacao_e_numeros():
    assert M.words("Olá, Mundo! 123 é-bom") == ["olá", "mundo", "é", "bom"]


def test_distinct_n():
    assert M.distinct_n(["a a a a"], 1) == pytest.approx(0.25)
    assert M.distinct_n(["a b c d"], 1) == 1.0
    assert M.distinct_n(["a b a b"], 2) == pytest.approx(2 / 3)
    assert M.distinct_n([""], 1) == 0.0


def test_invented_word_rate():
    vocab = {"uma", "noite"}
    assert M.invented_word_rate(["uma noite xyz"], vocab) == pytest.approx(1 / 3)
    assert M.invented_word_rate([""], vocab) == 0.0


def test_copy_rate():
    ref = M.corpus_ngrams(["uma noite destas vindo da cidade"], 3)
    assert M.copy_rate(["uma noite destas"], ref, 3) == 1.0
    assert M.copy_rate(["uma noite diferente"], ref, 3) == 0.0
    assert M.copy_rate(["uma noite"], ref, 3) == 0.0  # curto demais para ter 3-grama


def test_replacement_char_rate():
    assert M.replacement_char_rate(["ok", "ruim \ufffd"]) == 0.5
    assert M.replacement_char_rate([]) == 0.0


def test_bytes_por_token(tok):
    n = M.token_byte_lengths(tok)
    assert len(n) == tok.vocab_size
    assert int(n[tok.pad_id]) == int(n[tok.bos_id]) == int(n[tok.eos_id]) == 0
    assert int(n[3 + ord("a")]) == 1
    assert int(n.max()) > 1  # existem tokens que juntam vários bytes


# ------------------------------------------------------- métricas do modelo
def test_modelo_uniforme_tem_loss_ln_vocab(tok):
    ids = torch.randint(3, tok.vocab_size, (1000,))
    r = M.evaluate_split(ModeloUniforme(tok.vocab_size), ids, tok, context_length=32)
    assert r["loss"] == pytest.approx(math.log(tok.vocab_size), rel=1e-5)
    assert r["perplexity"] == pytest.approx(tok.vocab_size, rel=1e-4)
    assert r["tokens"] == len(range(0, 1000 - 32, 32)) * 32


def test_bits_por_byte_confere_com_a_conta_manual(tok):
    ids = torch.tensor(tok.encode(CORPUS)[:600])
    r = M.evaluate_split(ModeloUniforme(tok.vocab_size), ids, tok, context_length=32)
    n = len(range(0, 600 - 32, 32))
    byte_len = M.token_byte_lengths(tok)
    alvo_bytes = sum(int(byte_len[ids[s + 1 : s + 33]].sum()) for s in range(0, 600 - 32, 32))
    esperado = (n * 32 * math.log(tok.vocab_size)) / math.log(2) / alvo_bytes
    assert r["bits_per_byte"] == pytest.approx(esperado, rel=1e-5)


def test_avaliacao_e_deterministica(tok):
    torch.manual_seed(0)
    model = LyraModel(
        LyraConfig(vocab_size=tok.vocab_size, context_length=32, d_model=32,
                   n_layers=2, n_heads=4, d_ff=64, dropout=0.1)
    )
    model.train()  # mesmo em modo treino, a avaliação usa eval() por dentro
    ids = torch.tensor(tok.encode(CORPUS)[:800])
    a = M.evaluate_split(model, ids, tok, 32)
    b = M.evaluate_split(model, ids, tok, 32)
    assert a == b
    assert model.training  # e devolve o modo original


def test_sequencia_curta_demais(tok):
    with pytest.raises(ValueError):
        M.evaluate_split(ModeloUniforme(tok.vocab_size), torch.arange(10), tok, 32)


# ---------------------------------------------------------------- comparação
def resultado(**override):
    base = {
        "val": {"loss": 4.0, "perplexity": 55.0, "bits_per_byte": 1.2},
        "test": {"loss": 4.5, "perplexity": 90.0, "bits_per_byte": 1.3},
        "generation": {
            "distinct_1": 0.3, "distinct_2": 0.8, "invented_word_rate": 0.1,
            "replacement_char_rate": 0.2, "copy_rate_6": 0.05,
        },
        "chat": {"turn_marker_leaks": 0, "empty_reply_rate": 0.0, "prompt_within_budget": True},
        "name": "x",
    }
    for caminho, valor in override.items():
        secao, chave = caminho.split(".")
        base[secao][chave] = valor
    return base


def status(rows, metrica):
    return next(r["status"] for r in rows if r["metric"] == metrica)


def test_compare_melhorou_piorou_igual():
    rows = compare(
        resultado(),
        resultado(**{"val.loss": 3.5, "generation.distinct_1": 0.2, "test.loss": 4.5}),
    )
    assert status(rows, "val.loss") == "improved"      # menor é melhor
    assert status(rows, "generation.distinct_1") == "worse"  # maior é melhor
    assert status(rows, "test.loss") == "same"


def test_compare_booleano_e_informativo():
    rows = compare(
        resultado(),
        resultado(**{"chat.prompt_within_budget": False, "generation.copy_rate_6": 0.5}),
    )
    assert status(rows, "chat.prompt_within_budget") == "worse"
    assert status(rows, "generation.copy_rate_6") == "info"


# --------------------------------------------------------------- ponta a ponta
def test_run_evaluation_completo(tok):
    torch.manual_seed(0)
    cfg = LyraConfig(vocab_size=tok.vocab_size, context_length=128, d_model=32,
                n_layers=2, n_heads=4, d_ff=64, dropout=0.0)
    engine = LyraEngine(LyraModel(cfg), tok)
    dataset = TokenDataset(tok.encode(CORPUS * 3), 0.1, 0.1)
    prompts = {"completion_prompts": ["Lyra é", "Usuário"], "chat_script": ["olá", "tudo bem?"]}
    persona = Persona(name="Lyra")
    kwargs = dict(engine=engine, dataset=dataset, corpus_texts=[CORPUS],
                prompts_config=prompts, persona=persona, name="teste")

    a = run_evaluation(**kwargs)
    b = run_evaluation(**kwargs)
    assert a == b  # reprodutível
    json.dumps(a)  # serializável
    for chave in ["val", "test", "generation", "chat", "parameters", "vocab_size"]:
        assert chave in a
    assert a["chat"]["prompt_within_budget"] is True
    assert a["generation"]["num_prompts"] == 2