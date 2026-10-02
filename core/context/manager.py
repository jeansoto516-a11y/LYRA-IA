"""Gerenciador de contexto: monta o prompt dentro do limite de tokens do modelo.

O modelo atual NÃO foi treinado com um formato de chat. O formato abaixo usa só
texto comum (sem tokens especiais novos), então é compatível com o checkpoint atual.
"""
from dataclasses import dataclass

from core.context.messages import Message


@dataclass
class BuiltContext:
    token_ids: list[int]
    text: str
    included_messages: int
    dropped_messages: int
    truncated_last: bool


class ContextManager:
    def __init__(
        self,
        tokenizer,
        context_length: int,
        reserve_tokens: int,
        system_text: str,
        user_label: str = "Usuário",
        assistant_label: str = "Lyra",
    ) -> None:
        self.tokenizer = tokenizer
        self.reserve_tokens = reserve_tokens
        self.budget = context_length - reserve_tokens
        if self.budget <= 0:
            raise ValueError("reserve_tokens precisa ser menor que o contexto do modelo")
        self.system_text = " ".join(system_text.split())
        self.user_label = user_label
        self.assistant_label = assistant_label

        system_len = len(tokenizer.encode(self.system_text))
        if system_len > self.budget * 3 // 5:
            raise ValueError(
                f"Instruções do sistema têm {system_len} tokens; o máximo é "
                f"{self.budget * 3 // 5} (60% do espaço disponível)."
            )

    @staticmethod
    def _clean(text: str) -> str:
        # Une todas as quebras de linha e espaços: o texto do usuário não consegue
        # criar uma linha nova e, assim, não consegue forjar o início de um turno.
        return " ".join(text.split())

    def _label(self, role: str) -> str:
        return self.user_label if role == "user" else self.assistant_label

    def render(self, messages: list[Message]) -> str:
        lines = [f"{self._label(m.role)}: {self._clean(m.content)}" for m in messages]
        return (
            self.system_text
            + "\n\n"
            + "\n".join(lines)
            + f"\n{self.assistant_label}:"
        )

    def build(self, history: list[Message]) -> BuiltContext:
        if not history:
            raise ValueError("O histórico está vazio")

        best = None
        for n in range(1, len(history) + 1):
            text = self.render(history[-n:])
            ids = self.tokenizer.encode(text)
            if len(ids) > self.budget:
                break
            best = (text, ids, n)

        if best is not None:
            text, ids, n = best
            return BuiltContext(ids, text, n, len(history) - n, False)

        # Nem a última mensagem sozinha cabe: corta o começo dela até caber.
        last = history[-1]
        content = self._clean(last.content)
        while content:
            content = content[max(1, len(content) // 10) :]
            text = self.render([Message(last.role, content)])
            ids = self.tokenizer.encode(text)
            if len(ids) <= self.budget:
                break
        else:
            text = self.render([Message(last.role, "")])
            ids = self.tokenizer.encode(text)
        return BuiltContext(ids, text, 1, len(history) - 1, True)