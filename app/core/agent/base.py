from abc import ABC, abstractmethod


class ToolBase(ABC):
    """Interface for agent tools.

    Every tool has a name and description — these are fed to the LLM so it can
    decide which tool to call for a given question.

    Platform-specific tools (Magento, BigCommerce) live in app/agents/<platform>/.
    Generic tools (RAG search) live in app/core/agent/tools/.
    """

    name: str
    description: str

    @abstractmethod
    async def execute(self, input: dict) -> dict: ...
