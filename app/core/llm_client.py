from abc import ABC, abstractmethod


class LLMClientBase(ABC):
    """Interface for LLM provider backends.

    All backends must implement generate() and stream().
    Services depend only on this interface — never on a concrete provider.
    Swapping providers (Anthropic → OpenAI, Gemini, etc.) requires
    only a new implementation and a config change.

    generate() return dict keys: answer (str), tokens_used (int), model (str)
    stream()   yields: str tokens, then a final dict with tokens_used and model
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        model_override: str | None = None,
        system: str | None = None,
    ) -> dict: ...

    @abstractmethod
    async def stream(self, prompt: str, max_tokens: int = 1000): ...
