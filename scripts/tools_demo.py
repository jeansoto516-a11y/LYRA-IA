"""Demonstração do sistema de ferramentas.

O 'modelo' aqui é SIMULADO (respostas combinadas): o modelo atual da Lyra ainda não
sabe pedir ferramentas. O que a demonstração prova é a camada de ferramentas:
validação, permissões, confirmação humana, limites de pasta e auditoria.
"""
import json
import re

from core.context.messages import Message
from tools.agent import ToolAgent
from tools.audit import AuditLog
from tools.builtin import register_builtin_tools
from tools.executor import ToolExecutor
from tools.permissions import Principal
from tools.protocol import describe_tools_for_prompt
from tools.registry import ToolRegistry
from tools.schema import ParamSpec, Tool

AVISOS: list[str] = []


def enviar_aviso(mensagem: str):
    AVISOS.append(mensagem)
    return {"enviado": True, "total_de_avisos": len(AVISOS)}


def chamada(name: str, **arguments) -> str:
    return f'<tool_call>{json.dumps({"name": name, "arguments": arguments})}</tool_call>'


def dados_da_ferramenta(msgs) -> dict:
    """Lê o JSON do último resultado de ferramenta que o 'modelo' recebeu."""
    texto = msgs[-1].content
    corpo = texto.split(">", 1)[1].rsplit("</tool_result", 1)[0]
    return json.loads(corpo)


def modelo_simulado(*etapas):
    fila = list(etapas)

    def responder(msgs):
        etapa = fila.pop(0)
        return etapa(msgs) if callable(etapa) else etapa

    return responder


def confirmar(tool, args) -> bool:
    resposta = input(f"  >> Confirmar '{tool.name}' com {args}? (s/n): ").strip().lower()
    return resposta == "s"


def main() -> None:
    registry = ToolRegistry()
    register_builtin_tools(registry, files_dir="docs")
    registry.register(
        Tool(
            name="enviar_aviso",
            description="Envia um aviso para a equipe.",
            handler=enviar_aviso,
            data_access="nenhum",
            actions="registra um aviso para a equipe",
            parameters=(ParamSpec("mensagem", "string", max_length=200),),
            required_permissions=frozenset({"notices:send"}),
            sensitive=True,
        )
    )
    audit = AuditLog()
    executor = ToolExecutor(registry, audit)

    jean = Principal("jean", frozenset({"math:use", "time:read", "files:read", "notices:send"}))
    visitante = Principal("visitante", frozenset())

    print("Ferramentas que o Jean pode usar:\n" + describe_tools_for_prompt(registry, jean))
    print("\nFerramentas que o visitante pode usar:\n" + describe_tools_for_prompt(registry, visitante))

    def cenario(titulo, principal, pergunta, responder, confirm=None):
        print(f"\n=== {titulo} ===")
        print(f"Usuário ({principal.user_id}): {pergunta}")
        agent = ToolAgent(responder, executor, principal, confirm=confirm)
        r = agent.run([Message("user", pergunta)])
        for passo in r.steps:
            print(f"  [ferramenta] {passo.call.name}({passo.call.arguments}) -> {passo.result.status}")
        print(f"Lyra (modelo simulado) [{r.stopped}]: {r.text}")

    cenario(
        "1. Calcular", jean, "Quanto é (12 + 8) * 3?",
        modelo_simulado(
            chamada("calculate", expression="(12 + 8) * 3"),
            lambda m: f"O resultado é {dados_da_ferramenta(m)['output']['result']}.",
        ),
    )
    cenario(
        "2. Ler um arquivo da pasta docs", jean, "O que tem no ROADMAP.md?",
        modelo_simulado(
            chamada("read_text_file", path="ROADMAP.md"),
            lambda m: f"Li {len(dados_da_ferramenta(m)['output']['content'])} caracteres do ROADMAP.md.",
        ),
    )
    cenario(
        "3. Tentativa de sair da pasta autorizada", jean, "Leia ../README.md",
        modelo_simulado(
            chamada("read_text_file", path="../README.md"),
            lambda m: "Não consegui ler esse arquivo: " + dados_da_ferramenta(m)["error"] + ".",
        ),
    )
    cenario(
        "4. Usuário sem permissão", visitante, "Quanto é 2 + 2?",
        modelo_simulado(
            chamada("calculate", expression="2 + 2"),
            lambda m: "Não tenho permissão para calcular para este usuário.",
        ),
    )
    cenario(
        "5. Ferramenta sensível (pede sua confirmação)", jean, "Avise a equipe da reunião.",
        modelo_simulado(
            chamada("enviar_aviso", mensagem="Reunião hoje às 15h."),
            lambda m: "Aviso enviado." if dados_da_ferramenta(m)["output"] else "Não enviei o aviso.",
        ),
        confirm=confirmar,
    )

    print("\n=== Registro de auditoria ===")
    for e in audit.entries:
        print(f"  {e['user']:<9} {e['tool']:<16} {e['status']:<18} {e['duration_ms']:>7.1f} ms")
    print(f"\nAvisos realmente enviados: {AVISOS}")


if __name__ == "__main__":
    main()