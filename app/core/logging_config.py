import logging
import structlog


def configure_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Configure structlog for the application.

    log_format="json"    — machine-readable, for production / log aggregators
    log_format="console" — human-readable, for local development
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if log_format == "console":
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Keep stdlib logging in sync (used by uvicorn, httpx, etc.)
    logging.basicConfig(level=level, format="%(message)s")


def configure_logging_from_settings() -> None:
    from app.dependencies import get_settings
    s = get_settings()
    configure_logging(log_level=s.log_level, log_format=s.log_format)


def get_logger(name: str = "retriv"):
    return structlog.get_logger(name)
