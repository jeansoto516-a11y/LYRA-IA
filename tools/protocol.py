"""Protocolo entre o modelo e as ferramentas.

O modelo pede uma ferramenta escrevendo, em texto:
    <tool_call>{"name": "calculate", "arguments": {"expression": "2+2"}}</tool_call>

Tudo que o modelo escreve é tratado como entrada NÃO confiável: aqui só se extrai a
chamada; quem valida e autoriza é o ToolExecutor. O resultado volta para o modelo
dentro de <tool_result>, com '<' e '>' escapados, para que nenhum texto vindo de uma
ferramenta (por exemplo uma página da web) consiga forjar uma nova chamada.
"""
import json
from dataclasses import dataclass

from tools.executor import ToolCall, ToolResult
from tools.schema import NAME_PATTERN

OPEN = "<tool_call>"
CLOSE = "</tool_call>"
MAX_CALL_CHARS = 2000


@dataclass
class ParsedOutput:
    text: str  # texto visível que o modelo escreveu antes da chamada
    call: ToolCall | None = None
    error: str | None = None
    ignored_calls: int = 0  # só a primeira chamada de cada resposta é considerada


def parse_model_output(text: str) -> ParsedOutput:
    start = text.find(OPEN)
    if start == -1:
        return ParsedOutput(text.strip())

    visible = text[:start].strip()
    rest = text[start + len(OPEN):]
    end = rest.find(CLOSE)
    if end == -1:
        return ParsedOutput(visible, error="chamada incompleta: falta </tool_call>")

    body = rest[:end].strip()
    ignored = rest[end + len(CLOSE):].count(OPEN)
    if len(body) > MAX_CALL_CHARS:
        return ParsedOutput(visible, error="chamada grande demais", ignored_calls=ignored)

    try:
        data = json.loads(body)
    except (ValueError, RecursionError):
        return ParsedOutput(visible, error="JSON inválido na chamada", ignored_calls=ignored)

    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        return ParsedOutput(visible, error="a chamada precisa ter 'name' (texto)",
                            ignored_calls=ignored)
    extra = set(data) - {"name", "arguments"}
    if extra:
        return ParsedOutput(visible, error=f"campos não permitidos: {sorted(extra)}",
                            ignored_calls=ignored)
    arguments = data.get("arguments", {})
    if not isinstance(arguments, dict):
        return ParsedOutput(visible, error="'arguments' precisa ser um objeto",
                            ignored_calls=ignored)
    return ParsedOutput(visible, ToolCall(data["name"], arguments), None, ignored)


def _escape(text: str) -> str:
    return text.replace("<", "\\u003c").replace(">", "\\u003e")


def render_tool_result(result: ToolResult) -> str:
    """Texto que volta para o modelo. É dado, nunca instrução."""
    name = result.tool if NAME_PATTERN.match(result.tool or "") else "desconhecida"
    body = _escape(json.dumps({"output": result.output, "error": result.error},
                              ensure_ascii=False))
    return f'<tool_result name="{name}" status="{result.status}">{body}</tool_result>'


def render_protocol_error(message: str) -> str:
    body = _escape(json.dumps({"output": None, "error": message}, ensure_ascii=False))
    return f'<tool_result name="protocolo" status="invalid">{body}</tool_result>'


def describe_tools_for_prompt(registry, principal) -> str:
    """Lista curta das ferramentas que ESTE usuário pode usar, para pôr no prompt."""
    lines = [
        "Ferramentas disponíveis. Para usar uma, escreva "
        f'{OPEN}{{"name": "...", "arguments": {{...}}}}{CLOSE}.'
    ]
    for tool in registry.available_for(principal):
        params = ", ".join(
            f"{p.name}{'' if p.required else '?'}: {p.type}" for p in tool.parameters
        )
        sensitive = " (exige confirmação)" if tool.sensitive else ""
        lines.append(f"- {tool.name}({params}): {tool.description}{sensitive}")
    return "\n".join(lines)