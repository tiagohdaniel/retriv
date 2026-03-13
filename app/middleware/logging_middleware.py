import time
import uuid

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger("retriv.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Adds a unique request ID to every request and logs timing + status.

    - Reads X-Request-ID from incoming headers if present (enables correlation
      with upstream services / load balancers).
    - Generates a new UUID when no header is present.
    - Propagates the ID via structlog context so all log lines within a request
      share the same request_id field.
    - Adds X-Request-ID to the response headers.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response
