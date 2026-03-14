from anthropic import AsyncAnthropic

from app.core.llm_client import LLMClientBase


_SYSTEM_PROMPT = """You are a documentation assistant. Your sole purpose is to answer questions based on the provided documentation excerpts.

Rules you must follow without exception:
1. Answer ONLY using information present in the provided documentation. Do not use any external knowledge.
2. If the question cannot be answered from the documentation, respond with: "Não encontrei informações sobre isso na base de conhecimento."
3. If the question is completely unrelated to the documentation (e.g. general knowledge, trivia, coding help), respond with: "Só consigo responder perguntas relacionadas aos documentos indexados."
4. Never answer questions about topics not covered in the documentation, even if you know the answer.
5. Be concise and cite the source document when relevant."""


class AnthropicClient(LLMClientBase):
    """Anthropic Claude implementation of LLMClientBase."""

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-20250514", timeout: float = 30.0):
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key, timeout=timeout)

    async def generate(self, prompt: str, max_tokens: int = 1000) -> dict:
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "answer": message.content[0].text,
            "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
            "model": self.model,
        }

    async def stream(self, prompt: str, max_tokens: int = 1000):
        """Yields str tokens, then a final dict with tokens_used and model."""
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
            message = await stream.get_final_message()
            yield {
                "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
                "model": self.model,
            }
