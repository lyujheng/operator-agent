"""Retry logic with exponential backoff and jitter.

Used by LLMClient for transient HTTP errors (429, 5xx, network).
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Callable, TypeVar

import httpx

from agent_framework.exceptions import (
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Errors that warrant a retry.
_RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    """Return True if *exc* is transient and worth retrying."""
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.WriteError)):
        return True
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_HTTP_STATUS
    return False


async def retry_with_backoff(
    fn: Callable[..., Any],
    *,
    max_retries: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 4.0,
    multiplier: float = 2.0,
    jitter_ratio: float = 0.125,
) -> Any:
    """Call *fn* (an async callable) with exponential backoff on retryable errors.

    Raises the last exception if all retries are exhausted.
    Non-retryable errors propagate immediately.
    """
    delay = initial_delay
    last_exc: Exception | None = None

    for attempt in range(1 + max_retries):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc

            if not _is_retryable(exc):
                _raise_typed(exc)

            if attempt >= max_retries:
                _raise_typed(exc)

            # Apply jitter: delay * (1 ± jitter_ratio)
            jittered = delay * (1 + random.uniform(-jitter_ratio, jitter_ratio))
            logger.warning(
                "Retry %d/%d after %.2fs: %s",
                attempt + 1,
                max_retries,
                jittered,
                exc,
            )
            await asyncio.sleep(jittered)
            delay = min(delay * multiplier, max_delay)

    # Should be unreachable, but satisfy type checker.
    assert last_exc is not None
    _raise_typed(last_exc)


def _raise_typed(exc: Exception) -> None:
    """Re-raise *exc* as the appropriate framework exception type."""
    if isinstance(exc, httpx.TimeoutException):
        raise LLMTimeoutError(str(exc), context={"original": type(exc).__name__}) from exc
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.WriteError)):
        raise LLMConnectionError(str(exc), context={"original": type(exc).__name__}) from exc
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        raise LLMResponseError(
            f"HTTP {status}: {exc.response.text[:500]}",
            context={"status_code": status},
        ) from exc
    # Unknown exception — propagate as-is.
    raise exc
