import contextlib
import json
import math
import time
from pathlib import Path

import torch

from core.model.lyra import LyraModel
from training.checkpoint import load_checkpoint, save_checkpoint
from training.config import TrainConfig
from training.dataset import TokenDataset, load_tokens


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Aquecimento linear e depois decaimento em cosseno até min_lr."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


@torch.no_grad()
def evaluate(model, dataset, split, cfg: TrainConfig, device: str) -> float:
    """Média da loss em batches fixos (mesma semente), para comparar entre avaliações."""
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = dataset.get_batch(split, cfg.batch_size, cfg.model.context_length, gen)
        _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    model.train(was_training)
    return sum(losses) / len(losses)


def train(cfg: TrainConfig, resume: bool = False, stop_at: int | None = None) -> dict:
    """Treina o modelo. stop_at pausa o treino num passo (o cronograma de lr segue max_steps)."""
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)

    tokens_path = Path(cfg.tokens_path)
    ids = load_tokens(tokens_path)
    meta_path = tokens_path.with_suffix(".json")
    dataset_meta = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    if "vocab_size" in dataset_meta and dataset_meta["vocab_size"] != cfg.model.vocab_size:
        print(
            f"Aviso: vocab_size do modelo ({cfg.model.vocab_size}) ajustado para "
            f"{dataset_meta['vocab_size']} (o valor do tokenizer)."
        )
        cfg.model.vocab_size = dataset_meta["vocab_size"]

    dataset = TokenDataset(ids, cfg.val_fraction, cfg.test_fraction)
    print(f"Tokens por split: {dataset.sizes()}")

    model = LyraModel(cfg.model).to(device)
    print(f"Modelo: {model.num_parameters():,} parâmetros | dispositivo: {device}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
    )

    ckpt_dir = Path(cfg.checkpoint_dir)
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"
    log_path = ckpt_dir / "metrics.jsonl"

    batch_gen = torch.Generator().manual_seed(cfg.seed)
    step = 0
    best_val = float("inf")
    bad_evals = 0

    if resume and last_path.exists():
        ckpt = load_checkpoint(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        batch_gen.set_state(ckpt["generator_state"])
        step = ckpt["step"]
        best_val = ckpt["best_val_loss"]
        bad_evals = ckpt["bad_evals"]
        print(f"Retomando do passo {step} (melhor val loss até agora: {best_val:.4f})")
    elif resume:
        print("Nenhum checkpoint para retomar; começando do zero.")

    use_amp = cfg.mixed_precision and device == "cuda"
    amp_ctx = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if use_amp
        else contextlib.nullcontext
    )

    def build_state(val_loss: float) -> dict:
        return {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "generator_state": batch_gen.get_state(),
            "step": step,
            "best_val_loss": best_val,
            "bad_evals": bad_evals,
            "val_loss": val_loss,
            "model_config": cfg.model.to_dict(),
            "train_config": cfg.to_dict(),
            "seed": cfg.seed,
            "dataset_meta": dataset_meta,
            "torch_version": str(torch.__version__),
        }

    model.train()
    start = time.time()
    stopped_early = False

    while step < cfg.max_steps and (stop_at is None or step < stop_at):
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = dataset.get_batch(
                "train", cfg.batch_size, cfg.model.context_length, batch_gen
            )
            with amp_ctx():
                _, loss = model(x.to(device), y.to(device))
            (loss / cfg.grad_accum_steps).backward()
            train_loss += loss.item() / cfg.grad_accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        step += 1

        if step % cfg.log_interval == 0:
            elapsed = time.time() - start
            print(
                f"passo {step}/{cfg.max_steps} | loss {train_loss:.4f} | "
                f"lr {lr:.2e} | grad {float(grad_norm):.2f} | {elapsed:.0f}s"
            )

        if step % cfg.eval_interval == 0 or step == cfg.max_steps or step == stop_at:
            val_loss = evaluate(model, dataset, "val", cfg, device)
            improved = val_loss < best_val - cfg.min_delta
            if improved:
                best_val = val_loss
                bad_evals = 0
            else:
                bad_evals += 1

            record = {
                "step": step,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "lr": lr,
                "best": improved,
            }
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            marca = " (melhor)" if improved else ""
            print(f"  avaliação: val loss {val_loss:.4f}{marca}")

            state = build_state(val_loss)
            save_checkpoint(last_path, state)
            if improved:
                save_checkpoint(best_path, state)

            if bad_evals >= cfg.patience:
                print(f"Parada antecipada: sem melhora há {bad_evals} avaliações.")
                stopped_early = True
                break

    summary = {
        "steps": step,
        "best_val_loss": best_val,
        "stopped_early": stopped_early,
        "test_loss": None,
    }

    if best_path.exists():
        best = load_checkpoint(best_path, map_location=device)
        model.load_state_dict(best["model_state"])
        test_loss = evaluate(model, dataset, "test", cfg, device)
        summary["test_loss"] = test_loss
        print(
            f"\nMelhor val loss: {best_val:.4f} | test loss: {test_loss:.4f} "
            f"(perplexidade de teste: {math.exp(test_loss):.1f})"
        )
    return summary