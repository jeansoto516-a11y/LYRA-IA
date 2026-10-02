"""Persona da Lyra: configurável por arquivo, sem alterar os pesos do modelo."""
import json
from dataclasses import dataclass, field
from pathlib import Path


def _join_pt(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " e " + items[-1]


@dataclass
class Persona:
    name: str = "Lyra"
    user_label: str = "Usuário"
    role: str = "assistente"
    traits: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)

    def system_text(self) -> str:
        intro = f"{self.name} é uma {self.role}"
        if self.traits:
            intro += " " + _join_pt(self.traits)
        parts = [intro + "."] + list(self.rules)
        return " ".join(parts)

    @classmethod
    def from_dict(cls, data: dict) -> "Persona":
        return cls(**data)

    @classmethod
    def from_json(cls, path) -> "Persona":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))