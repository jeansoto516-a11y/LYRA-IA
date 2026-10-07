"""Registro estruturado (JSON por linha) de tudo que as ferramentas fazem."""
import json
import time
from pathlib import Path

MAX_MEMORY_ENTRIES = 1000


class AuditLog:
    def __init__(self, path=None, clock=time.time) -> None:
        self.path = Path(path) if path else None
        self.clock = clock
        self.entries: list[dict] = []

    def record(self, **fields) -> dict:
        entry = {"ts": round(self.clock(), 3), **fields}
        self.entries.append(entry)
        if len(self.entries) > MAX_MEMORY_ENTRIES:
            del self.entries[: len(self.entries) - MAX_MEMORY_ENTRIES]
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry