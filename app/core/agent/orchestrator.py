from app.core.agent.base import ToolBase
from app.core.llm_client import LLMClientBase


class AgentOrchestrator:
    """Generic agentic loop — platform-agnostic.

    Holds a registry of tools and exposes their definitions to the LLM.
    The full agentic loop (LLM decides which tool → execute → feed result back → repeat)
    will be activated when LLMClientBase gains tool_use() support in a future feature.

    To add a new platform agent:
    - Create app/agents/<platform>/tools.py with tools extending ToolBase
    - Register them here via the tools parameter
    - Core orchestration logic does not change
    """

    MAX_ITERATIONS = 10

    def __init__(self, tools: list[ToolBase], llm_client: LLMClientBase):
        self.tools: dict[str, ToolBase] = {tool.name: tool for tool in tools}
        self.llm = llm_client

    def get_tool_definitions(self) -> list[dict]:
        """Returns tool definitions for LLM tool use (name + description)."""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools.values()
        ]

    async def run_tool(self, tool_name: str, input: dict) -> dict:
        """Executes a single tool by name."""
        tool = self.tools.get(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found."}
        return await tool.execute(input)
