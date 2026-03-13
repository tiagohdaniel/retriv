from app.core.agent.orchestrator import AgentOrchestrator


class AgentService:
    """Entry point for agent-based interactions.

    Currently routes through RagTool only — same behavior as AskService.
    Designed to support multi-tool orchestration as new tools are registered.

    When LLMClientBase gains tool_use() support, this service will drive the
    full agentic loop: LLM selects tool → execute → feed result → repeat.
    """

    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator

    def available_tools(self) -> list[dict]:
        return self.orchestrator.get_tool_definitions()
