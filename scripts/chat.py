import argparse

from conversation.session import ChatSession
from core.inference.engine import LyraEngine
from personality.persona import Persona

def main() -> None: 
    parser = argparse.ArgumentParser(description="Conversa com a Lyra no terminal")
    parser.add_argument("--checkpoint", default="checkpoints/lyra_0_1/best.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer/lyra_tokenizer_v0.json")
    parser.add_argument("--persona", default="configs/persona_lyra.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    engine = LyraEngine.from_checkpoint(args.checkpoint, args.tokenizer)
    persona = Persona.from_json(args.persona)
    session = ChatSession(engine, persona)

    print(f"{persona.name} (modelo base, ainda sem treino de conversa).")
    print("Comandos: /limpar apaga o histórico, /sair encerra.\n")

    while True:
        try:
            text = input(f"{persona.user_label}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/sair":
            break
        if text =="/limpar":
            session.reset()
            print("(historico apagado)\n")
            continue
        repley = session.send(text, temperature=args.temperature)
        print(f"{persona.name}: {repley}\n")

if __name__ == "__main__":
    main()