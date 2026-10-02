import argparse

from core.inference.engine import LyraEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera texto com a Lyra")
    parser.add_argument("--prompt", default="Uma noite destas")
    parser.add_argument("--checkpoint", default="checkpoints/lyra_0_1/best.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer/lyra_tokenizer_v0.json")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    engine = LyraEngine.from_checkpoint(args.checkpoint, args.tokenizer)
    text = engine.generate_text(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(text)


if __name__ == "__main__":
    main()