from pathlib import Path

from core.tokenizer.bpe import BPETokenizer

CORPUS_FILE = Path("data/processed/dom_casmurro_clean.txt")
OUTPUT_FILE = Path("data/tokenizer/lyra_tokenizer_v0.json")
VOCAB_SIZE = 1500


def main() -> None:
    text = CORPUS_FILE.read_text(encoding="utf-8")
    print(f"Corpus: {len(text)} caracteres")

    tok = BPETokenizer()
    print(f"Treinando tokenizer (vocabulário de {VOCAB_SIZE} tokens)...")
    tok.train(text, vocab_size=VOCAB_SIZE)
    tok.save(OUTPUT_FILE)
    print(f"Salvo em: {OUTPUT_FILE}")

    # Teste rápido: ida e volta (texto -> ids -> texto)
    exemplo = "Uma noite destas, vindo da cidade para o Engenho Novo, encontrei um rapaz."
    ids = tok.encode(exemplo)
    voltou = tok.decode(ids)
    print(f"\nTexto:   {exemplo}")
    print(f"Tokens:  {len(ids)} (contra {len(exemplo.encode('utf-8'))} bytes)")
    print(f"IDs:     {ids}")
    print(f"Decodif: {voltou}")
    print(f"Ida e volta correta? {voltou == exemplo}")


if __name__ == "__main__":
    main()