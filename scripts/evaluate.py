import argparse
import json
from pathlib import Path

import torch

from core.inference.engine import LyraEngine
from evaluation.runner import compare, run_evaluation
from personality.persona import Persona
from training.dataset import TokenDataset, load_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia um checkpoint da Lyra")
    parser.add_argument("--name", default="lyra-0.1")
    parser.add_argument("--checkpoint", default="checkpoints/lyra_0_1/best.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer/lyra_tokenizer_v0.json")
    parser.add_argument("--tokens", default="data/processed/tokens.npy")
    parser.add_argument("--corpus-glob", default="data/processed/*_clean.txt")
    parser.add_argument("--prompts", default="configs/eval_prompts.json")
    parser.add_argument("--persona", default="configs/persona_lyra.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--compare", default=None, help="JSON de uma avaliação anterior")
    args = parser.parse_args()

    output = Path(args.output or f"evaluation/results/{args.name}.json")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    train_cfg = ckpt.get("train_config", {})
    dataset = TokenDataset(
        load_tokens(args.tokens),
        train_cfg.get("val_fraction", 0.05),
        train_cfg.get("test_fraction", 0.05),
    )

    engine = LyraEngine.from_checkpoint(args.checkpoint, args.tokenizer)
    corpus_texts = [
        p.read_text(encoding="utf-8") for p in sorted(Path().glob(args.corpus_glob))
    ]
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    persona = Persona.from_json(args.persona)

    print("Avaliando (leva cerca de 1 minuto)...")
    result = run_evaluation(
        engine, dataset, corpus_texts, prompts, persona, args.name, step=ckpt.get("step")
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== {result['name']} (passo {result['checkpoint_step']}) ===")
    print(f"Parâmetros: {result['parameters']:,} | vocabulário: {result['vocab_size']}")
    for split in ("val", "test"):
        r = result[split]
        print(
            f"{split:>4}: loss {r['loss']:.4f} | perplexidade {r['perplexity']:.1f} | "
            f"bits/byte {r['bits_per_byte']:.3f} ({r['tokens']} tokens)"
        )
    g = result["generation"]
    print(
        f"geração: distinct-1 {g['distinct_1']:.2f} | distinct-2 {g['distinct_2']:.2f} | "
        f"palavras inventadas {g['invented_word_rate']:.1%} | "
        f"cópia do corpus (6-gramas) {g['copy_rate_6']:.1%} | "
        f"textos com bytes inválidos {g['replacement_char_rate']:.0%}"
    )
    c = result["chat"]
    print(
        f"chat: vazamentos de turno {c['turn_marker_leaks']} | "
        f"respostas vazias {c['empty_reply_rate']:.0%} | "
        f"prompt dentro do limite: {'sim' if c['prompt_within_budget'] else 'NÃO'}"
    )
    print(f"\nSalvo em: {output}")

    if args.compare:
        old = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        print(f"\nComparação com {old['name']}:")
        for row in compare(old, result):
            print(
                f"  {row['metric']:<36} {str(row['old'])[:8]:>9} -> "
                f"{str(row['new'])[:8]:<9} {row['status']}"
            )


if __name__ == "__main__":
    main()