import pytest

from core.tokenizer.bpe import BASE_VOCAB_SIZE, BPETokenizer

CORPUS = (
    "Uma noite destas, vindo da cidade para o Engenho Novo, encontrei no trem "
    "da Central um rapaz aqui do bairro, que eu conheço de vista e de chapéu. "
    "A coração não tem razão, mas a ação é de São Paulo em 1857. "
) * 50


@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=400, verbose=False)
    return t


TEXTOS = [
    "Olá, mundo!",
    "Uma noite destas, vindo da cidade.",
    "coração ação São Paulo é à ü ñ",
    "Preço: R$ 1.234,56 em 2026 (10%)",
    "Veja https://exemplo.com.br/pagina?x=1&y=2",
    "def soma(a, b):\n    return a + b",
    "Jean Carlos Soto Barbosa",
    "palavraqueNaoExisteEmLugarNenhumxyzqq",
    "emoji 😀 e símbolos ∑ ≠ ©",
    "linha1\n\nlinha2\t\ttab",
    "   espaços   no   começo e no fim   ",
    "",
]


@pytest.mark.parametrize("texto", TEXTOS)
def test_ida_e_volta(tok, texto):
    assert tok.decode(tok.encode(texto)) == texto


def test_ids_dentro_do_vocabulario(tok):
    for texto in TEXTOS:
        for i in tok.encode(texto):
            assert 0 <= i < tok.vocab_size


def test_compressao_em_texto_do_corpus(tok):
    texto = "Uma noite destas, vindo da cidade para o Engenho Novo"
    assert len(tok.encode(texto)) < len(texto.encode("utf-8"))


def test_palavra_desconhecida_vira_bytes(tok):
    ids = tok.encode("zzqxj")
    assert len(ids) > 0
    assert tok.decode(ids) == "zzqxj"


def test_tokens_especiais(tok):
    ids = tok.encode("oi", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == "oi"
    assert len({tok.pad_id, tok.bos_id, tok.eos_id}) == 3


def test_tamanho_do_vocabulario(tok):
    assert BASE_VOCAB_SIZE <= tok.vocab_size <= 400


def test_treino_deterministico():
    a, b = BPETokenizer(), BPETokenizer()
    a.train(CORPUS, vocab_size=350, verbose=False)
    b.train(CORPUS, vocab_size=350, verbose=False)
    assert a.merges == b.merges


def test_salvar_e_carregar(tok, tmp_path):
    caminho = tmp_path / "tok.json"
    tok.save(caminho)
    carregado = BPETokenizer.load(caminho)
    assert carregado.merges == tok.merges
    for texto in TEXTOS:
        assert carregado.encode(texto) == tok.encode(texto)


def test_vocab_size_invalido():
    with pytest.raises(ValueError):
        BPETokenizer().train("abc", vocab_size=10, verbose=False)