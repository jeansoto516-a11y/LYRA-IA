from tools.schema import Tool


class ToolRegistry:
    """Catálogo de ferramentas. O núcleo da Lyra não conhece ferramentas específicas:
    cada sistema registra as suas aqui."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("Registre objetos do tipo Tool")
        if tool.name in self._tools:
            raise ValueError(f"Já existe uma ferramenta chamada {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def available_for(self, principal) -> list[Tool]:
        """Só as ferramentas que esse usuário tem permissão de usar."""
        return [t for _, t in sorted(self._tools.items()) if principal.can_use(t)]

    def describe(self, principal=None) -> list[dict]:
        tools = self.available_for(principal) if principal else [
            t for _, t in sorted(self._tools.items())
        ]
        return [t.to_schema() for t in tools]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)