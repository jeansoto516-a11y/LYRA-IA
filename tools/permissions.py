"""Permissões: princípio do menor privilégio.

Uma permissão tem o formato "área:ação" (por exemplo "files:read"). Não existe
curinga: cada ferramenta pede exatamente o que precisa e quem chama (Principal)
recebe exatamente o que foi concedido pelo sistema anfitrião.

O Principal também carrega o tenant (a organização do usuário, por exemplo uma
imobiliária) e o papel. Os dois vêm SEMPRE da sessão do sistema anfitrião,
nunca dos argumentos escritos pelo modelo.
"""
import re
from dataclasses import dataclass
from typing import Optional

PERMISSION_PATTERN = re.compile(r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$")
TENANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
ROLE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def validate_permissions(perms) -> frozenset:
    perms = frozenset(perms)
    for p in perms:
        if not isinstance(p, str) or not PERMISSION_PATTERN.match(p):
            raise ValueError(f"Permissão inválida: {p!r} (use o formato 'area:acao')")
    return perms


def validate_tenant_id(tenant_id: Optional[str]) -> Optional[str]:
    if tenant_id is None:
        return None
    if not isinstance(tenant_id, str) or not TENANT_ID_PATTERN.match(tenant_id):
        raise ValueError(
            f"tenant_id inválido: {tenant_id!r} "
            "(use de 1 a 64 caracteres: letras, números, '_' ou '-')"
        )
    return tenant_id


def validate_role(role: Optional[str]) -> Optional[str]:
    if role is None:
        return None
    if not isinstance(role, str) or not ROLE_PATTERN.match(role):
        raise ValueError(
            f"Papel inválido: {role!r} (use letras minúsculas, números e '_')"
        )
    return role


@dataclass(frozen=True)
class Principal:
    """Quem está pedindo a execução. Criado pelo sistema anfitrião, nunca pelo modelo."""

    user_id: str
    permissions: frozenset = frozenset()
    tenant_id: Optional[str] = None
    role: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, str) or not self.user_id.strip():
            raise ValueError("user_id não pode ser vazio")
        object.__setattr__(self, "permissions", validate_permissions(self.permissions))
        object.__setattr__(self, "tenant_id", validate_tenant_id(self.tenant_id))
        object.__setattr__(self, "role", validate_role(self.role))

    def missing_permissions(self, tool) -> frozenset:
        return frozenset(tool.required_permissions) - self.permissions

    def can_use(self, tool) -> bool:
        return not self.missing_permissions(tool)

    def same_tenant(self, other_tenant_id: Optional[str]) -> bool:
        """True só se os dois tenants existem e são idênticos."""
        return self.tenant_id is not None and self.tenant_id == other_tenant_id