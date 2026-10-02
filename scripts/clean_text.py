import re
from pathlib import Path

START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"

INPUT_FILE = Path("data/raw/dom_casmurro.txt")
OUTPUT_FILE = Path("data/processed/dom_casmurro_clean.txt")


def clean_gutenberg(text: str) -> str:
    # Remove o cabeçalho (tudo até a linha do START_MARKER)
    start = text.find(START_MARKER)
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0

    # Remove o rodapé (tudo a partir do END_MARKER)
    end = text.find(END_MARKER)
    if end == -1:
        end = len(text)

    body = text[start:end]

    # Normaliza quebras de linha e espaços
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)

    return body.strip() + "\n"


def main() -> None:
    raw = INPUT_FILE.read_text(encoding="utf-8-sig")
    clean = clean_gutenberg(raw)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(clean, encoding="utf-8")

    print(f"Caracteres antes:  {len(raw)}")
    print(f"Caracteres depois: {len(clean)}")
    print("--- INÍCIO ---")
    print(clean[:300])
    print("--- FIM ---")
    print(clean[-200:])


if __name__ == "__main__":
    main()