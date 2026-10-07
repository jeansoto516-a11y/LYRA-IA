"""Ferramentas básicas e seguras que acompanham a Lyra."""
import ast
import math
import operator
import re
from datetime import datetime, timezone
from pathlib import Path

from tools.schema import ParamSpec, Tool, ToolValidationError

# ------------------------------------------------------------------ calculate
MAX_EXPRESSION_CHARS = 200
MAX_POWER_EXPONENT = 100
MAX_POWER_BASE = 1e6
MAX_RESULT_BITS = 4000

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _eval_node(node):
    """Avalia só números e operações aritméticas. Nada de nomes, chamadas ou atributos."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_POWER_EXPONENT or abs(left) > MAX_POWER_BASE:
                raise ToolValidationError("potência grande demais")
            return left**right
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ToolValidationError("operação não permitida")
        return op(left, right)
    raise ToolValidationError("expressão não permitida (use só números e + - * / // % **)")


def calculate(expression: str):
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ToolValidationError("expressão longa demais")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree)
    except ZeroDivisionError:
        raise ToolValidationError("divisão por zero")
    except OverflowError:
        raise ToolValidationError("resultado grande demais")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise ToolValidationError("expressão inválida")
    if isinstance(result, float) and not math.isfinite(result):
        raise ToolValidationError("resultado não é um número finito")
    if isinstance(result, int) and result.bit_length() > MAX_RESULT_BITS:
        raise ToolValidationError("resultado grande demais")
    return {"result": result}


CALCULATE_TOOL = Tool(
    name="calculate",
    description="Calcula uma expressão matemática simples.",
    handler=calculate,
    data_access="nenhum",
    actions="nenhuma (só calcula)",
    parameters=(
        ParamSpec("expression", "string", "ex.: (12 + 8) * 3", max_length=MAX_EXPRESSION_CHARS),
    ),
    required_permissions=frozenset({"math:use"}),
    timeout_seconds=2.0,
)


# ------------------------------------------------------------ get_current_time
def get_current_time():
    return {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


TIME_TOOL = Tool(
    name="get_current_time",
    description="Informa a data e a hora atuais em UTC.",
    handler=get_current_time,
    data_access="relógio do sistema",
    actions="nenhuma",
    required_permissions=frozenset({"time:read"}),
    timeout_seconds=1.0,
)


# --------------------------------------------------------------- read_text_file
ALLOWED_EXTENSIONS = (".txt", ".md", ".json", ".csv")
_DENIED = "arquivo não encontrado ou não permitido"  # mesma mensagem para todos os casos


def make_read_text_file_tool(base_dir, max_bytes: int = 20000) -> Tool:
    """Lê arquivos de texto SOMENTE dentro de base_dir."""
    base = Path(base_dir).resolve()

    def read_text_file(path: str):
        path = path.replace("\\", "/")  # trata \ e / do mesmo jeito em qualquer sistema
        if (
            not path.strip()
            or ".." in path.split("/")
            or path.startswith("/")
            or re.match(r"^[A-Za-z]:", path)
        ):
            raise ToolValidationError(_DENIED)
        try:
            candidate = (base / path).resolve(strict=True)
            inside = candidate.is_relative_to(base)
        except (OSError, ValueError):
            raise ToolValidationError(_DENIED)
        if (
            not inside
            or not candidate.is_file()
            or candidate.suffix.lower() not in ALLOWED_EXTENSIONS
        ):
            raise ToolValidationError(_DENIED)
        if candidate.stat().st_size > max_bytes:
            raise ToolValidationError(f"arquivo maior que {max_bytes} bytes")
        text = candidate.read_bytes().decode("utf-8", errors="replace")
        return {"path": candidate.relative_to(base).as_posix(), "content": text}

    return Tool(
        name="read_text_file",
        description="Lê um arquivo de texto da pasta autorizada.",
        handler=read_text_file,
        data_access=f"arquivos {', '.join(ALLOWED_EXTENSIONS)} dentro de uma única pasta autorizada",
        actions="nenhuma (somente leitura)",
        parameters=(ParamSpec("path", "string", "caminho relativo à pasta autorizada", max_length=300),),
        required_permissions=frozenset({"files:read"}),
        timeout_seconds=3.0,
    )


def register_builtin_tools(registry, files_dir=None) -> None:
    registry.register(CALCULATE_TOOL)
    registry.register(TIME_TOOL)
    if files_dir is not None:
        registry.register(make_read_text_file_tool(files_dir))