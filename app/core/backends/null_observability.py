from app.core.observability import ObservabilityBase


class NullObservability(ObservabilityBase):
    """No-op — used when eval_enabled=false."""

    async def trace_query(self, **kwargs) -> None:
        pass
