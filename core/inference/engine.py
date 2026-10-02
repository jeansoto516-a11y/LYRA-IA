"""Camada de inferência: carrega um checkpoint e gera texto.

Não depende dos scripts de treinamento.
"""
import torch

from core.generation.sampling import generate
from core.model.config import LyraConfig
from core.model.lyra import LyraModel
from core.tokenizer.bpe import BPETokenizer


class LyraEngine:
    def __init__(self, model: LyraModel, tokenizer: BPETokenizer, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint_path, tokenizer_path, device: str = "cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = LyraModel(LyraConfig(**ckpt["model_config"]))
        model.load_state_dict(ckpt["model_state"])
        tokenizer = BPETokenizer.load(tokenizer_path)
        if tokenizer.vocab_size != model.config.vocab_size:
            raise ValueError(
                f"Tokenizer ({tokenizer.vocab_size}) e modelo "
                f"({model.config.vocab_size}) têm vocabulários diferentes."
            )
        return cls(model, tokenizer, device)

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int | None = 40,
        top_p: float | None = 0.95,
    ) -> str:
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.tokenizer.bos_id]
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        out = generate(
            self.model,
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_ids=[self.tokenizer.eos_id],
        )
        ids = [i for i in out[0].tolist() if i != self.tokenizer.eos_id]
        return self.tokenizer.decode(ids)