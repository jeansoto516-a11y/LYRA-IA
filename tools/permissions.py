"""Permissões: princípio do menor privilégio.

Uma permissão tem o formato "área:ação" (por exemplo "files:read"). Não existe
curinga: cada ferramenta pede exatamente o que precisa e quem chama (Principal)
recebe exatamente o que foi concedido pelo sistema anfitrião.
"""
import re
from dataclasses import dataclass

PERMISSION_PATTERN = re.compile(r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$")


def validate_permissions(perms) -> frozenset:
    perms = frozenset(perms)
    for p in perms:
        if not isinstance(p, str) or not PERMISSION_PATTERN.match(p):
            raise ValueError(f"Permissão inválida: {p!r} (use o formato 'area:acao')")
    return perms


@dataclass(frozen=True)
class Principal:
    """Quem está pedindo a execução. Criado pelo sistema anfitrião, nunca pelo modelo."""

    user_id: str
    permissions: frozenset = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, str) or not self.user_id.strip():
            raise ValueError("user_id não pode ser vazio")
        object.__setattr__(self, "permissions", validate_permissions(self.permissions))

    def missing_permissions(self, tool) -> frozenset:
        return frozenset(tool.required_permissions) - self.permissions

    def can_use(self, tool) -> bool:
        return not self.missing_permissions(tool)