"""Execução segura de ferramentas.

Ordem de cada chamada: existe? -> o usuário pode usar? -> argumentos válidos? ->
precisa de confirmação humana? -> executa com tempo limite -> limita a saída -> registra.
Tudo que vem do modelo é tratado como não confiável.
"""
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

from tools.audit import AuditLog
from tools.registry import ToolRegistry
from tools.schema import ToolValidationError, validate_arguments

STATUSES = (
    "ok", "error", "denied", "invalid", "not_found",
    "needs_confirmation", "rejected", "timeout",
)
MAX_LOGGED_VALUE = 200


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)
    call_id: str = ""


@dataclass
class ToolResult:
    call_id: str
    tool: str
    status: str
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id, "tool": self.tool, "status": self.status,
            "output": self.output, "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
        }


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        audit: AuditLog | None = None,
        max_output_chars: int = 2000,
        confirm: Callable | None = None,
    ) -> None:
        self.registry = registry
        self.audit = audit or AuditLog()
        self.max_output_chars = max_output_chars
        self.confirm = confirm

    # ------------------------------------------------------------ auxiliares
    @staticmethod
    def _safe_args(tool, arguments) -> dict:
        """Argumentos para o log: segredos e chaves desconhecidas nunca aparecem."""
        if not isinstance(arguments, dict):
            return {}
        specs = {p.name: p for p in tool.parameters} if tool else {}
        safe = {}
        for key, value in arguments.items():
            spec = specs.get(key)
            if spec is None or spec.secret:
                safe[str(key)] = "***"
            elif isinstance(value, str) and len(value) > MAX_LOGGED_VALUE:
                safe[str(key)] = value[:MAX_LOGGED_VALUE] + "..."
            else:
                safe[str(key)] = value
        return safe

    def _limit_output(self, output):
        try:
            text = json.dumps(output, ensure_ascii=False)
        except (TypeError, ValueError):
            raise ToolValidationError("A saída da ferramenta não é serializável")
        if len(text) <= self.max_output_chars:
            return output
        return text[: self.max_output_chars] + "...[truncado]"

    def _finish(self, call, tool, principal, status, started, output=None, error=None):
        duration_ms = (time.perf_counter() - started) * 1000
        result = ToolResult(call.call_id, call.name, status, output, error, duration_ms)
        self.audit.record(
            call_id=call.call_id,
            tool=call.name,
            user=principal.user_id,
            status=status,
            duration_ms=round(duration_ms, 2),
            arguments=self._safe_args(tool, call.arguments),
            error=error,
            output_chars=len(json.dumps(output, ensure_ascii=False)) if output is not None else 0,
        )
        return result

    # --------------------------------------------------------------- execução
    def execute(self, call: ToolCall, principal, confirm: Callable | None = None) -> ToolResult:
        started = time.perf_counter()
        if not call.call_id:
            call = ToolCall(call.name, call.arguments, uuid.uuid4().hex[:8])

        tool = self.registry.get(call.name)
        if tool is None:
            return self._finish(call, None, principal, "not_found", started,
                                error="Ferramenta não encontrada")

        if not principal.can_use(tool):
            return self._finish(call, tool, principal, "denied", started,
                                error="Permissão insuficiente para esta ferramenta")

        try:
            arguments = validate_arguments(tool, call.arguments)
        except ToolValidationError as e:
            return self._finish(call, tool, principal, "invalid", started, error=str(e))

        if tool.sensitive:
            decide = confirm or self.confirm
            if decide is None:
                return self._finish(call, tool, principal, "needs_confirmation", started,
                                    error="Esta ferramenta exige confirmação humana")
            if not decide(tool, arguments):
                return self._finish(call, tool, principal, "rejected", started,
                                    error="Execução recusada na confirmação")

        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(tool.handler, **arguments)
        try:
            raw = future.result(timeout=tool.timeout_seconds)
            output = self._limit_output(raw)
        except FutureTimeout:
            return self._finish(call, tool, principal, "timeout", started,
                                error=f"Passou do tempo limite de {tool.timeout_seconds}s")
        except ToolValidationError as e:
            return self._finish(call, tool, principal, "error", started, error=str(e))
        except Exception as e:  # a mensagem original pode conter dados sensíveis
            return self._finish(call, tool, principal, "error", started,
                                error=f"A ferramenta falhou ({type(e).__name__})")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        return self._finish(call, tool, principal, "ok", started, output=output)