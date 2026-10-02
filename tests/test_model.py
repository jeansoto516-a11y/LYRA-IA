import math

import pytest
import torch

from core.generation.sampling import generate
from core.model.config import LyraConfig
from core.model.lyra import LyraModel

VOCAB = 300


def make_model() -> LyraModel:
    torch.manual_seed(0)
    config = LyraConfig(
        vocab_size=VOCAB,
        context_length=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    )
    return LyraModel(config)


@pytest.fixture(scope="module")
def model():
    return make_model().eval()


def test_forma_da_saida(model):
    idx = torch.randint(0, VOCAB, (2, 10))
    logits, loss = model(idx)
    assert logits.shape == (2, 10, VOCAB)
    assert loss is None


def test_loss_inicial_perto_do_acaso(model):
    idx = torch.randint(0, VOCAB, (4, 12))
    targets = torch.randint(0, VOCAB, (4, 12))
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(VOCAB)) < 1.0


def test_causalidade(model):
    x1 = torch.randint(0, VOCAB, (1, 10))
    x2 = x1.clone()
    x2[:, -1] = (x2[:, -1] + 1) % VOCAB
    l1, _ = model(x1)
    l2, _ = model(x2)
    assert torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-5)


def test_sequencia_maior_que_contexto(model):
    with pytest.raises(ValueError):
        model(torch.randint(0, VOCAB, (1, 17)))


def test_numero_de_parametros_padrao():
    n = LyraModel(LyraConfig()).num_parameters()
    assert 900_000 < n < 1_100_000


def test_geracao_tamanho(model):
    prompt = torch.randint(0, VOCAB, (1, 3))
    out = generate(model, prompt, max_new_tokens=5)
    assert out.shape == (1, 8)


def test_geracao_alem_do_contexto(model):
    prompt = torch.randint(0, VOCAB, (1, 10))
    out = generate(model, prompt, max_new_tokens=20)
    assert out.shape == (1, 30)


def test_greedy_deterministico(model):
    prompt = torch.randint(0, VOCAB, (1, 4))
    a = generate(model, prompt, max_new_tokens=6, temperature=0)
    b = generate(model, prompt, max_new_tokens=6, temperature=0)
    assert torch.equal(a, b)


def test_top_k_1_equivale_a_greedy(model):
    prompt = torch.randint(0, VOCAB, (1, 4))
    greedy = generate(model, prompt, max_new_tokens=6, temperature=0)
    topk1 = generate(model, prompt, max_new_tokens=6, temperature=1.0, top_k=1)
    assert torch.equal(greedy, topk1)


def test_top_p_gera_ids_validos(model):
    prompt = torch.randint(0, VOCAB, (1, 4))
    out = generate(model, prompt, max_new_tokens=10, top_p=0.9)
    assert out.shape == (1, 14)
    assert int(out.min()) >= 0 and int(out.max()) < VOCAB


def test_stop_token_interrompe(model):
    prompt = torch.randint(0, VOCAB, (1, 4))
    primeiro = generate(model, prompt, max_new_tokens=1, temperature=0)
    stop_id = int(primeiro[0, -1])
    out = generate(
        model, prompt, max_new_tokens=10, temperature=0, stop_ids=[stop_id]
    )
    assert out.shape == (1, 5)


def test_modelo_consegue_aprender():
    m = make_model()
    m.train()
    seq = torch.arange(17).unsqueeze(0)
    x, y = seq[:, :-1], seq[:, 1:]
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    _, first = m(x, y)
    first = first.item()
    for _ in range(100):
        opt.zero_grad()
        _, loss = m(x, y)
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5