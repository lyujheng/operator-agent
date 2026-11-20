"""Context variable management for span nesting.

Uses :mod:`contextvars` so that ``trace()`` and ``span()`` context managers
automatically track parent/child relationships across async tasks.
"""

from __future__ import annotations

import logging
import traceback as tb_mod
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Generator

from agent_framework.tracing.trace import Span, Trace

if TYPE_CHECKING:
    from agent_framework.tracing.processors import TracingProcessor

logger = logging.getLogger(__name__)

# ── context variables ────────────────────────────────────────────

_current_trace: ContextVar[Trace | None] = ContextVar("_current_trace", default=None)
_current_span: ContextVar[Span | None] = ContextVar("_current_span", default=None)

# Global processor list — populated via ``register_processor``.
_processors: list[TracingProcessor] = []


def register_processor(processor: TracingProcessor) -> None:
    """Add a tracing processor to the global pipeline."""
    _processors.append(processor)


def clear_processors() -> None:
    """Remove all registered processors."""
    _processors.clear()


def get_current_trace() -> Trace | None:
    return _current_trace.get()


def get_current_span() -> Span | None:
    return _current_span.get()


# ── helpers ──────────────────────────────────────────────────────

def _notify_start(kind: str, obj: Trace | Span) -> None:
    for p in _processors:
        try:
            if kind == "trace":
                p.on_trace_start(obj)  # type: ignore[arg-type]
            else:
                p.on_span_start(obj)  # type: ignore[arg-type]
        except Exception:
            logger.debug("Processor %s failed on %s start", type(p).__name__, kind, exc_info=True)


def _notify_end(kind: str, obj: Trace | Span) -> None:
    for p in _processors:
        try:
            if kind == "trace":
                p.on_trace_end(obj)  # type: ignore[arg-type]
            else:
                p.on_span_end(obj)  # type: ignore[arg-type]
        except Exception:
            logger.debug("Processor %s failed on %s end", type(p).__name__, kind, exc_info=True)


# ── public context managers ──────────────────────────────────────

@contextmanager
def trace(name: str, **metadata: Any) -> Generator[Trace, None, None]:
    """Open a new :class:`Trace` scope.

    Usage::

        with trace("requirement_allocation", input_file="reqs.xlsx") as t:
            ...  # spans created inside automatically belong to *t*
    """
    t = Trace(name=name, metadata=metadata)
    token = _current_trace.set(t)
    _notify_start("trace", t)
    try:
        yield t
        t.finish(status="completed")
    except Exception as exc:
        t.finish(status="error")
        t.metadata["error"] = str(exc)
        raise
    finally:
        _notify_end("trace", t)
        _current_trace.reset(token)


@contextmanager
def span(
    name: str,
    *,
    span_type: str = "step",
    input_data: dict[str, Any] | None = None,
    **metadata: Any,
) -> Generator[Span, None, None]:
    """Open a new :class:`Span` within the current trace.

    Automatically sets *parent_id* to the enclosing span (if any) and
    registers the span on the active trace.
    """
    current_trace = _current_trace.get()
    parent = _current_span.get()

    s = Span(
        name=name,
        span_type=span_type,
        parent_id=parent.span_id if parent else None,
        input_data=input_data,
        metadata=metadata,
    )

    if current_trace is not None:
        current_trace.add_span(s)

    token = _current_span.set(s)
    _notify_start("span", s)
    try:
        yield s
        if s.status == "running":
            s.finish(status="completed")
    except Exception as exc:
        s.fail(message=str(exc), traceback=tb_mod.format_exc())
        raise
    finally:
        _notify_end("span", s)
        _current_span.reset(token)
