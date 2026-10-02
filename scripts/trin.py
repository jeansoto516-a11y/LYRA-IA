import argparse

from training.config import TrainConfig
from training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina a Lyra")
    parser.add_argument("--config", default="configs/train_v0.json")
    parser.add_argument("--resume", action="store_true", help="retoma do last.pt")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = TrainConfig.from_json(args.config)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()