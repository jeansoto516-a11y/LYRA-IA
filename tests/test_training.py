import numpy as np
import pytest
import torch

from core.inference.engine import LyraEngine
from core.model.config import LyraConfig
from core.tokenizer.bpe import BPETokenizer
from training.checkpoint import load_checkpoint, save_checkpoint
from training.config import TrainConfig
from training.trainer import get_lr, train

VOCAB = 40


@pytest.fixture
def tiny_cfg(tmp_path):
    # Sequência repetitiva: fácil de aprender, então a loss tem que cair
    ids = np.tile(np.arange(3, 3 + 20), 400).astype(np.int32)
    np.save(tmp_path / "tokens.npy", ids)
    return TrainConfig(
        model=LyraConfig(
            vocab_size=VOCAB, context_length=16, d_model=32, n_layers=2,
            n_heads=4, d_ff=64, dropout=0.0,
        ),
        tokens_path=str(tmp_path / "tokens.npy"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        batch_size=8,
        grad_accum_steps=2,
        max_steps=60,
        learning_rate=3e-3,
        min_lr=3e-4,
        warmup_steps=5,
        eval_interval=20,
        eval_batches=3,
        log_interval=1000,
        patience=100,
        device="cpu",
    )


def test_lr_schedule():
    cfg = TrainConfig(max_steps=100, warmup_steps=10, learning_rate=1e-3, min_lr=1e-4)
    assert get_lr(0, cfg) < get_lr(9, cfg) <= 1e-3
    assert get_lr(10, cfg) == pytest.approx(1e-3)
    assert get_lr(55, cfg) < get_lr(10, cfg)
    assert get_lr(100, cfg) == pytest.approx(1e-4)


def test_config_json_roundtrip(tmp_path):
    cfg = TrainConfig(max_steps=123)
    import json
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    assert TrainConfig.from_json(p).max_steps == 123


def test_config_rejeita_campo_desconhecido():
    with pytest.raises(TypeError):
        TrainConfig.from_dict({"campo_que_nao_existe": 1})


def test_checkpoint_atomico_ida_e_volta(tmp_path):
    state = {"step": 5, "w": torch.ones(3)}
    save_checkpoint(tmp_path / "a" / "x.pt", state)
    got = load_checkpoint(tmp_path / "a" / "x.pt")
    assert got["step"] == 5 and torch.equal(got["w"], torch.ones(3))
    assert not (tmp_path / "a" / "x.pt.tmp").exists()


def test_treino_reduz_loss_e_salva_checkpoints(tiny_cfg):
    resumo = train(tiny_cfg)
    ckpt_dir = tiny_cfg.checkpoint_dir
    from pathlib import Path
    assert (Path(ckpt_dir) / "best.pt").exists()
    assert (Path(ckpt_dir) / "last.pt").exists()
    assert (Path(ckpt_dir) / "metrics.jsonl").exists()
    assert resumo["steps"] == 60
    assert resumo["best_val_loss"] < 2.0  # acaso seria ln(40) ~ 3.7
    assert resumo["test_loss"] is not None


def test_checkpoint_guarda_metadados(tiny_cfg):
    train(tiny_cfg)
    from pathlib import Path
    ckpt = load_checkpoint(Path(tiny_cfg.checkpoint_dir) / "last.pt")
    for chave in ["model_state", "optimizer_state", "step", "model_config",
                  "train_config", "seed", "val_loss", "generator_state"]:
        assert chave in ckpt
    assert ckpt["seed"] == tiny_cfg.seed


def test_retomar_continua_do_passo_salvo(tiny_cfg):
    from pathlib import Path
    tiny_cfg.max_steps = 20
    train(tiny_cfg)
    assert load_checkpoint(Path(tiny_cfg.checkpoint_dir) / "last.pt")["step"] == 20
    tiny_cfg.max_steps = 40
    resumo = train(tiny_cfg, resume=True)
    assert resumo["steps"] == 40
    assert load_checkpoint(Path(tiny_cfg.checkpoint_dir) / "last.pt")["step"] == 40


def test_retomada_equivale_a_treino_continuo(tmp_path, tiny_cfg):
    from pathlib import Path

    tiny_cfg.max_steps = 40

    continuo = TrainConfig.from_dict(tiny_cfg.to_dict())
    continuo.checkpoint_dir = str(tmp_path / "continuo")
    train(continuo)

    pausado = TrainConfig.from_dict(tiny_cfg.to_dict())
    pausado.checkpoint_dir = str(tmp_path / "pausado")
    train(pausado, stop_at=20)
    assert load_checkpoint(Path(pausado.checkpoint_dir) / "last.pt")["step"] == 20
    train(pausado, resume=True)

    a = load_checkpoint(Path(continuo.checkpoint_dir) / "last.pt")["model_state"]
    b = load_checkpoint(Path(pausado.checkpoint_dir) / "last.pt")["model_state"]
    for chave in a:
        assert torch.allclose(a[chave], b[chave], atol=1e-6), chave


def test_parada_antecipada(tiny_cfg):
    tiny_cfg.patience = 1
    tiny_cfg.min_delta = 100.0  # nenhuma melhora conta
    tiny_cfg.max_steps = 200
    resumo = train(tiny_cfg)
    assert resumo["stopped_early"] is True
    assert resumo["steps"] < 200


def test_vocab_do_modelo_e_ajustado_ao_dataset(tiny_cfg, tmp_path):
    import json
    from pathlib import Path
    Path(tiny_cfg.tokens_path).with_suffix(".json").write_text(
        json.dumps({"vocab_size": VOCAB}), encoding="utf-8"
    )
    tiny_cfg.model.vocab_size = 999
    tiny_cfg.max_steps = 5
    tiny_cfg.eval_interval = 5
    train(tiny_cfg)
    assert tiny_cfg.model.vocab_size == VOCAB


def test_engine_gera_texto_a_partir_do_checkpoint(tmp_path):
    texto = "uma noite destas vindo da cidade " * 60
    tok = BPETokenizer()
    tok.train(texto, vocab_size=300, verbose=False)
    tok.save(tmp_path / "tok.json")

    ids = np.asarray(tok.encode(texto), dtype=np.int32)
    np.save(tmp_path / "tokens.npy", ids)
    cfg = TrainConfig(
        model=LyraConfig(
            vocab_size=tok.vocab_size, context_length=16, d_model=32, n_layers=2,
            n_heads=4, d_ff=64, dropout=0.0,
        ),
        tokens_path=str(tmp_path / "tokens.npy"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        batch_size=4, grad_accum_steps=1, max_steps=10, eval_interval=10,
        eval_batches=2, log_interval=1000, device="cpu",
    )
    train(cfg)
    engine = LyraEngine.from_checkpoint(
        tmp_path / "ckpt" / "best.pt", tmp_path / "tok.json"
    )
    saida = engine.generate_text("uma noite", max_new_tokens=8)
    assert saida.startswith("uma noite")
    assert len(saida) > len("uma noite")


def test_engine_rejeita_tokenizer_incompativel(tmp_path):
    texto = "uma noite destas vindo da cidade " * 60
    tok = BPETokenizer()
    tok.train(texto, vocab_size=300, verbose=False)
    tok.save(tmp_path / "tok.json")
    outro = BPETokenizer()
    outro.train(texto, vocab_size=280, verbose=False)
    outro.save(tmp_path / "outro.json")

    np.save(tmp_path / "tokens.npy", np.asarray(tok.encode(texto), dtype=np.int32))
    cfg = TrainConfig(
        model=LyraConfig(vocab_size=tok.vocab_size, context_length=16, d_model=32,
        n_layers=1, n_heads=4, d_ff=64, dropout=0.0),
        tokens_path=str(tmp_path / "tokens.npy"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        batch_size=4, grad_accum_steps=1, max_steps=5, eval_interval=5,
        eval_batches=2, log_interval=1000, device="cpu",
    )
    train(cfg)
    with pytest.raises(ValueError):
        LyraEngine.from_checkpoint(tmp_path / "ckpt" / "best.pt", tmp_path / "outro.json")