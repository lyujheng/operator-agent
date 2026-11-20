"""Framework unit tests - tracing."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agent_framework.tracing import (
    ConsoleProcessor,
    FileProcessor,
    MultiTracingProcessor,
    Span,
    Trace,
    clear_processors,
    export_trace_json,
    get_current_span,
    get_current_trace,
    register_processor,
    span,
    trace,
)


class TestSpanModel:
    def test_span_defaults(self):
        s = Span(name="test_op", span_type="tool")
        assert s.span_id.startswith("span_")
        assert s.status == "running"
        assert s.ended_at is None

    def test_span_finish(self):
        s = Span(name="test", span_type="step")
        s.finish()
        assert s.status == "completed"
        assert s.ended_at is not None

    def test_span_fail(self):
        s = Span(name="test", span_type="tool")
        s.fail("something broke", "traceback here")
        assert s.status == "error"
        assert s.error is not None
        assert s.error["message"] == "something broke"


class TestTraceModel:
    def test_trace_defaults(self):
        t = Trace()
        assert t.trace_id.startswith("trace_")
        assert t.status == "running"

    def test_add_span(self):
        t = Trace()
        s = Span(name="child", span_type="step")
        t.add_span(s)
        assert s.trace_id == t.trace_id
        assert len(t.spans) == 1


class TestContextManagers:
    def setup_method(self):
        clear_processors()

    def test_trace_context(self):
        with trace("test_trace") as t:
            assert get_current_trace() is t
        assert get_current_trace() is None

    def test_span_nesting(self):
        with trace("root") as t:
            with span("outer", span_type="step") as s1:
                assert get_current_span() is s1
                assert s1.parent_id is None  # top-level span
                with span("inner", span_type="tool") as s2:
                    assert get_current_span() is s2
                    assert s2.parent_id == s1.span_id
            assert get_current_span() is None
        assert len(t.spans) == 2

    def test_span_error_propagates(self):
        try:
            with trace("err_trace") as t:
                with span("failing", span_type="tool") as s:
                    raise ValueError("test error")
        except ValueError:
            pass
        assert s.status == "error"
        assert s.error is not None


class TestFileProcessor:
    def test_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = FileProcessor(tmpdir)
            s = Span(name="test_span", span_type="tool", trace_id="trace_abc123")
            s.finish()
            fp.on_span_end(s)
            fp.shutdown()

            log_file = Path(tmpdir) / "trace_abc123.jsonl"
            assert log_file.exists()
            lines = log_file.read_text().strip().split("\n")
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["name"] == "test_span"


class TestTraceExport:
    def test_export_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            t = Trace(name="export_test")
            s = Span(name="s1", span_type="step")
            t.add_span(s)
            s.finish()
            t.finish()

            path = export_trace_json(t, tmpdir)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["name"] == "export_test"
            assert len(data["spans"]) == 1
