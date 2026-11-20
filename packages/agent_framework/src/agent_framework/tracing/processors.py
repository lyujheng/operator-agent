"""TracingProcessors: ConsoleProcessor, FileProcessor, and multi-fan-out.

Each processor receives span/trace lifecycle events.  A failing processor
never blocks others (errors are logged, not propagated).
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO

from agent_framework.tracing.trace import Span, Trace

logger = logging.getLogger(__name__)


class TracingProcessor(ABC):
    """Interface for trace event consumers."""

    @abstractmethod
    def on_span_start(self, span: Span) -> None: ...

    @abstractmethod
    def on_span_end(self, span: Span) -> None: ...

    @abstractmethod
    def on_trace_start(self, trace: Trace) -> None: ...

    @abstractmethod
    def on_trace_end(self, trace: Trace) -> None: ...

    def shutdown(self) -> None:
        """Release resources.  Called once at process exit."""


# ── Console ──────────────────────────────────────────────────────

class ConsoleProcessor(TracingProcessor):
    """Prints human-readable log lines to the console."""

    def __init__(self, logger_name: str = "agent_framework.tracing") -> None:
        self._log = logging.getLogger(logger_name)

    def on_trace_start(self, trace: Trace) -> None:
        self._log.info("[%s] Trace started: %s", trace.trace_id, trace.name)

    def on_trace_end(self, trace: Trace) -> None:
        self._log.info(
            "[%s] Trace %s: %s (spans: %d)",
            trace.trace_id,
            trace.status,
            trace.name,
            len(trace.spans),
        )

    def on_span_start(self, span: Span) -> None:
        agent = span.metadata.get("agent_name", "-")
        self._log.info(
            "[%s] [%s] [%s] %s started",
            span.trace_id or "no-trace",
            agent,
            span.span_type,
            span.name,
        )

    def on_span_end(self, span: Span) -> None:
        agent = span.metadata.get("agent_name", "-")
        level = logging.WARNING if span.status == "error" else logging.INFO
        msg = f"[{span.trace_id or 'no-trace'}] [{agent}] [{span.span_type}] {span.name} {span.status}"
        if span.error:
            msg += f": {span.error.get('message', '')}"
        self._log.log(level, msg)

    def shutdown(self) -> None:
        pass


# ── File (JSONL) ─────────────────────────────────────────────────

class FileProcessor(TracingProcessor):
    """Appends each completed span as a JSON line to a ``.jsonl`` file."""

    def __init__(self, log_dir: str | Path) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, IO[str]] = {}

    def _get_handle(self, trace_id: str) -> IO[str]:
        if trace_id not in self._handles:
            path = self._log_dir / f"{trace_id}.jsonl"
            self._handles[trace_id] = open(path, "a", encoding="utf-8")  # noqa: SIM115
        return self._handles[trace_id]

    def on_trace_start(self, trace: Trace) -> None:
        pass  # file created lazily on first span end

    def on_trace_end(self, trace: Trace) -> None:
        pass  # handle closed in shutdown

    def on_span_start(self, span: Span) -> None:
        pass

    def on_span_end(self, span: Span) -> None:
        if not span.trace_id:
            return
        fh = self._get_handle(span.trace_id)
        line = span.model_dump_json()
        fh.write(line + "\n")
        fh.flush()

    def shutdown(self) -> None:
        for fh in self._handles.values():
            try:
                fh.close()
            except Exception:
                logger.debug("Failed to close log handle", exc_info=True)
        self._handles.clear()


# ── Multi-processor fan-out ──────────────────────────────────────

class MultiTracingProcessor(TracingProcessor):
    """Forwards events to multiple processors; one failure does not block others."""

    def __init__(self, processors: list[TracingProcessor] | None = None) -> None:
        self._processors = list(processors or [])

    def add(self, processor: TracingProcessor) -> None:
        self._processors.append(processor)

    def on_trace_start(self, trace: Trace) -> None:
        for p in self._processors:
            try:
                p.on_trace_start(trace)
            except Exception:
                logger.debug("Processor %s error on trace_start", type(p).__name__, exc_info=True)

    def on_trace_end(self, trace: Trace) -> None:
        for p in self._processors:
            try:
                p.on_trace_end(trace)
            except Exception:
                logger.debug("Processor %s error on trace_end", type(p).__name__, exc_info=True)

    def on_span_start(self, span: Span) -> None:
        for p in self._processors:
            try:
                p.on_span_start(span)
            except Exception:
                logger.debug("Processor %s error on span_start", type(p).__name__, exc_info=True)

    def on_span_end(self, span: Span) -> None:
        for p in self._processors:
            try:
                p.on_span_end(span)
            except Exception:
                logger.debug("Processor %s error on span_end", type(p).__name__, exc_info=True)

    def shutdown(self) -> None:
        for p in self._processors:
            try:
                p.shutdown()
            except Exception:
                logger.debug("Processor %s error on shutdown", type(p).__name__, exc_info=True)
