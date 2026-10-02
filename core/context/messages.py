from dataclasses import dataclass

ROLES = ("user", "assistant")


@dataclass(frozen=True)
class Message:
    """Uma mensagem da conversa (do usuário ou da Lyra)."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role inválido: {self.role!r} (use {ROLES})")
        if not isinstance(self.content, str):
            raise TypeError("content precisa ser texto")