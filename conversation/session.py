from core.context.manager import BuiltContext, ContextManager
from core.context.messages import Message
from core.inference.engine import LyraEngine
from personality.persona import Persona


class ChatSession:
    """Uma conversa: histórico + persona + gerenciador de contexto + motor de geração."""

    def __init__(self, engine: LyraEngine, persona: Persona, reserve_tokens: int = 40):
        self.engine = engine
        self.persona = persona
        self.history: list[Message] = []
        self.context = ContextManager(
            tokenizer=engine.tokenizer,
            context_length=engine.model.config.context_length,
            reserve_tokens=reserve_tokens,
            system_text=persona.system_text(),
            user_label=persona.user_label,
            assistant_label=persona.name,
        )
        self.last_context: BuiltContext | None = None

    def send(
        self,
        user_text: str,
        temperature: float = 0.7,
        top_k: int | None = 40,
        top_p: float | None = 0.9,
    ) -> str:
        self.history.append(Message("user", user_text))
        built = self.context.build(self.history)
        self.last_context = built

        stop = (f"\n{self.persona.user_label}:", f"\n{self.persona.name}:")
        reply = self.engine.complete(
            built.token_ids,
            max_new_tokens=self.context.reserve_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_strings=stop,
        ).strip()

        self.history.append(Message("assistant", reply))
        return reply

    def reset(self) -> None:
        self.history.clear()
        self.last_context = None