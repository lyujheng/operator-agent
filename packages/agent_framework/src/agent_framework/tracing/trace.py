"""Trace and Span pydantic models for structured observability."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _generate_id(prefix: str) -> str:
    """Generate a short prefixed hex id (e.g. ``span_a1b2c3d4e5f6``)."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class Span(BaseModel):
    """A single unit of work inside a trace.

    Spans nest via *parent_id* to form a tree.
    """

    span_id: str = Field(default_factory=lambda: _generate_id("span"))
    trace_id: str = ""
    parent_id: str | None = None
    name: str
    span_type: str  # "agent" | "tool" | "llm_call" | "step"
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ended_at: str | None = None
    status: str = "running"  # "running" | "completed" | "error"
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def finish(self, *, status: str = "completed") -> None:
        """Mark the span as finished with the current timestamp."""
        self.ended_at = datetime.now(timezone.utc).isoformat()
        self.status = status

    def fail(self, message: str, traceback: str = "") -> None:
        """Mark the span as failed and attach error details."""
        self.error = {"message": message, "traceback": traceback}
        self.finish(status="error")


class Trace(BaseModel):
    """Top-level container for a full execution run.

    A trace groups all spans produced during one requirement-processing session.
    """

    trace_id: str = Field(default_factory=lambda: _generate_id("trace"))
    name: str = "requirement_allocation"
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ended_at: str | None = None
    status: str = "running"  # "running" | "completed" | "error"
    metadata: dict[str, Any] = Field(default_factory=dict)
    spans: list[Span] = Field(default_factory=list)

    def add_span(self, span: Span) -> None:
        """Register a span under this trace, setting its trace_id."""
        span.trace_id = self.trace_id
        self.spans.append(span)

    def finish(self, *, status: str = "completed") -> None:
        self.ended_at = datetime.now(timezone.utc).isoformat()
        self.status = status
