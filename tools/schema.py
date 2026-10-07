"""Definição de ferramentas e validação dos argumentos."""
import math
import re
from dataclasses import dataclass
from typing import Any, Callable

from tools.permissions import validate_permissions

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
PARAM_TYPES = ("string", "integer", "number", "boolean")
DEFAULT_MAX_STRING = 1000


class ToolValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = None
    enum: tuple | None = None
    min_value: float | None = None
    max_value: float | None = None
    max_length: int = DEFAULT_MAX_STRING
    secret: bool = False  # se True, o valor nunca aparece nos logs

    def __post_init__(self) -> None:
        if not NAME_PATTERN.match(self.name):
            raise ValueError(f"Nome de parâmetro inválido: {self.name!r}")
        if self.type not in PARAM_TYPES:
            raise ValueError(f"Tipo inválido: {self.type!r} (use {PARAM_TYPES})")
        if self.enum is not None:
            object.__setattr__(self, "enum", tuple(self.enum))

    def to_schema(self) -> dict:
        schema = {"name": self.name, "type": self.type, "description": self.description,
                  "required": self.required}
        if self.enum is not None:
            schema["enum"] = list(self.enum)
        if self.min_value is not None:
            schema["min"] = self.min_value
        if self.max_value is not None:
            schema["max"] = self.max_value
        if self.type == "string":
            schema["max_length"] = self.max_length
        return schema


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]
    data_access: str  # que dados a ferramenta lê ("nenhum" se não lê nada)
    actions: str  # que ações a ferramenta executa ("nenhuma" se só consulta)
    parameters: tuple = ()
    required_permissions: frozenset = frozenset()
    sensitive: bool = False  # exige confirmação humana antes de executar
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not NAME_PATTERN.match(self.name):
            raise ValueError(f"Nome de ferramenta inválido: {self.name!r}")
        if not self.description.strip():
            raise ValueError("A ferramenta precisa de uma descrição")
        if not callable(self.handler):
            raise ValueError("handler precisa ser chamável")
        if not self.data_access.strip() or not self.actions.strip():
            raise ValueError("Declare data_access e actions (use 'nenhum'/'nenhuma' se for o caso)")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds precisa ser positivo")
        params = tuple(self.parameters)
        names = [p.name for p in params]
        if len(names) != len(set(names)):
            raise ValueError("Parâmetros com nomes repetidos")
        object.__setattr__(self, "parameters", params)
        object.__setattr__(
            self, "required_permissions", validate_permissions(self.required_permissions)
        )

    def to_schema(self) -> dict:
        """Descrição pública (sem o handler), segura para mostrar ou serializar."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_schema() for p in self.parameters],
            "required_permissions": sorted(self.required_permissions),
            "sensitive": self.sensitive,
            "data_access": self.data_access,
            "actions": self.actions,
            "timeout_seconds": self.timeout_seconds,
        }


def _check_type(spec: ParamSpec, value) -> str | None:
    if spec.type == "string":
        if not isinstance(value, str):
            return "precisa ser texto"
        if len(value) > spec.max_length:
            return f"passa do limite de {spec.max_length} caracteres"
    elif spec.type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return "precisa ser um número inteiro"
    elif spec.type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "precisa ser um número"
        if isinstance(value, float) and not math.isfinite(value):
            return "precisa ser um número finito"
    elif spec.type == "boolean":
        if not isinstance(value, bool):
            return "precisa ser verdadeiro ou falso"

    if spec.enum is not None and value not in spec.enum:
        return f"precisa ser um destes valores: {list(spec.enum)}"
    if spec.type in ("integer", "number"):
        if spec.min_value is not None and value < spec.min_value:
            return f"não pode ser menor que {spec.min_value}"
        if spec.max_value is not None and value > spec.max_value:
            return f"não pode ser maior que {spec.max_value}"
    return None


def validate_arguments(tool: Tool, arguments) -> dict:
    """Confere os argumentos contra o schema. Devolve só o que é válido (+ padrões)."""
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolValidationError("Os argumentos precisam ser um objeto (dicionário)")

    known = {p.name for p in tool.parameters}
    problems = []

    unknown = sorted(str(k) for k in arguments if k not in known)
    if unknown:
        problems.append(f"parâmetros desconhecidos: {unknown}")

    cleaned = {}
    for spec in tool.parameters:
        if spec.name not in arguments:
            if spec.required:
                problems.append(f"'{spec.name}' é obrigatório")
            elif spec.default is not None:
                cleaned[spec.name] = spec.default
            continue
        value = arguments[spec.name]
        error = _check_type(spec, value)
        if error:
            problems.append(f"'{spec.name}' {error}")
        else:
            cleaned[spec.name] = value

    if problems:
        raise ToolValidationError("; ".join(problems))
    return cleaned