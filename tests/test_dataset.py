import pytest
import torch

from core.tokenizer.bpe import BPETokenizer
from training.dataset import (
    TokenDataset,
    deduplicate_paragraphs,
    encode_documents,
    load_tokens,
    normalize_text,
    save_tokens,
)


@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer()
    t.train("uma noite destas vindo da cidade " * 40, vocab_size=300, verbose=False)
    return t


def test_normalize_nfc():
    decomposto = "coraça\u0303o"  # "a" + til combinante
    assert normalize_text(decomposto) == "coração"
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"


def test_deduplicacao_remove_paragrafo_longo_repetido():
    longo = "Este é um parágrafo bem longo que se repete no texto. " * 3
    texto = f"{longo}\n\noutro\n\n{longo}\n\noutro"
    resultado = deduplicate_paragraphs(texto)
    assert resultado.count(longo.strip()) == 1


def test_deduplicacao_preserva_paragrafos_curtos():
    texto = "—Sim.\n\n—Não.\n\n—Sim."
    assert deduplicate_paragraphs(texto) == texto


def test_encode_documents_tem_bos_e_eos(tok):
    ids = encode_documents(["uma noite", "da cidade"], tok)
    lista = ids.tolist()
    assert lista[0] == tok.bos_id
    assert lista.count(tok.bos_id) == 2
    assert lista.count(tok.eos_id) == 2
    assert lista[-1] == tok.eos_id


def test_salvar_e_carregar_tokens(tok, tmp_path):
    ids = encode_documents(["uma noite destas"], tok)
    caminho = tmp_path / "t.npy"
    save_tokens(ids, caminho, {"x": 1})
    assert load_tokens(caminho).tolist() == ids.tolist()
    assert (tmp_path / "t.json").exists()


def test_splits_somam_e_nao_se_sobrepoem():
    ds = TokenDataset(list(range(1000)), val_fraction=0.1, test_fraction=0.1)
    tamanhos = ds.sizes()
    assert sum(tamanhos.values()) == 1000
    assert tamanhos == {"train": 800, "val": 100, "test": 100}
    assert int(ds.splits["train"].max()) < int(ds.splits["val"].min())
    assert int(ds.splits["val"].max()) < int(ds.splits["test"].min())


def test_batch_formato_e_deslocamento():
    ds = TokenDataset(list(range(1000)))
    x, y = ds.get_batch("train", batch_size=4, block_size=16)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y, x + 1)  # dados são 0,1,2,... então o alvo é x+1


def test_batch_nunca_passa_do_fim():
    ds = TokenDataset(list(range(200)), val_fraction=0.1, test_fraction=0.1)
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        x, y = ds.get_batch("val", batch_size=8, block_size=10, generator=g)
        assert int(y.max()) <= int(ds.splits["val"].max())


def test_batch_com_gerador_e_reprodutivel():
    ds = TokenDataset(list(range(1000)))
    a = ds.get_batch("train", 4, 16, generator=torch.Generator().manual_seed(1))
    b = ds.get_batch("train", 4, 16, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a[0], b[0])


def test_split_pequeno_demais():
    ds = TokenDataset(list(range(100)), val_fraction=0.05, test_fraction=0.05)
    with pytest.raises(ValueError):
        ds.get_batch("val", batch_size=2, block_size=16)


def test_fracoes_invalidas():
    with pytest.raises(ValueError):
        TokenDataset(list(range(100)), val_fraction=0.6, test_fraction=0.5)