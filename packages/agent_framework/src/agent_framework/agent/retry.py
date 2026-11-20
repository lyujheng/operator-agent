"""RetryPolicy - unified retry configuration."""

from __future__ import annotations

from pydantic import BaseModel


class RetryPolicy(BaseModel):
    """Shared retry parameters injected across LLM / Tool / MCP layers."""

    max_retries: int = 3
    initial_delay: float = 0.5
    max_delay: float = 4.0
    multiplier: float = 2.0
    jitter_ratio: float = 0.125
    retryable_errors: list[str] = []  # error class names; empty → module default
