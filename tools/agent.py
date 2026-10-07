"""Laço de uso de ferramentas: o modelo pede, a Lyra executa, o resultado volta.

O 'responder' é qualquer coisa que transforme a lista de mensagens em texto: pode ser
o modelo da Lyra ou, nos testes, um ator que simula o modelo.
"""
from dataclasses import dataclass, field
from typing import Callable

from core.context.messages import Message
from tools.executor import ToolCall, ToolExecutor, ToolResult
from tools.protocol import parse_model_output, render_protocol_error, render_tool_result


@dataclass
class AgentStep:
    call: ToolCall
    result: ToolResult


@dataclass
class AgentResult:
    text: str
    steps: list = field(default_factory=list)
    stopped: str = "answered"  # answered | max_steps | needs_confirmation
    pending: ToolCall | None = None


class ToolAgent:
    def __init__(
        self,
        responder: Callable[[list[Message]], str],
        executor: ToolExecutor,
        principal,
        max_steps: int = 3,
        confirm: Callable | None = None,
    ) -> None:
        self.responder = responder
        self.executor = executor
        self.principal = principal
        self.max_steps = max_steps
        self.confirm = confirm

    def run(self, messages: list[Message]) -> AgentResult:
        history = list(messages)
        steps: list[AgentStep] = []

        for iteration in range(self.max_steps + 1):
            output = self.responder(history)
            parsed = parse_model_output(output)

            if parsed.error is None and parsed.call is None:
                return AgentResult(parsed.text, steps, "answered")

            if iteration == self.max_steps:
                # O modelo ainda quer outra ferramenta, mas o limite de passos acabou.
                return AgentResult(parsed.text, steps, "max_steps")

            history.append(Message("assistant", output))

            if parsed.error is not None:
                history.append(Message("tool", render_protocol_error(parsed.error)))
                continue

            result = self.executor.execute(parsed.call, self.principal, confirm=self.confirm)
            steps.append(AgentStep(parsed.call, result))
            if result.status == "needs_confirmation":
                return AgentResult(parsed.text, steps, "needs_confirmation", parsed.call)
            history.append(Message("tool", render_tool_result(result)))

        return AgentResult("", steps, "max_steps")  # não deve chegar aqui