"""Framework unit tests - ReAct agent."""

from __future__ import annotations

import json
import pytest

from agent_framework.agent.react import ReActAgent
from agent_framework.agent.tool_registry import ToolRegistry, func_to_schema
from agent_framework.agent.base import truncate_tool_result, MessageHistory
from agent_framework.llm.types import Message
from agent_framework.exceptions import MaxTurnsExceeded


# ── Tool function tests ──────────────────────────────────────────

class TestFuncToSchema:
    def test_simple_function(self):
        def greet(name: str, times: int = 1) -> str:
            """Say hello."""
            return f"Hello {name}" * times

        schema = func_to_schema(greet)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "greet"
        assert "name" in schema["function"]["parameters"]["properties"]
        assert "name" in schema["function"]["parameters"]["required"]
        assert "times" not in schema["function"]["parameters"]["required"]

    def test_no_annotations(self):
        def mystery(x, y):
            """Do something."""
            pass

        schema = func_to_schema(mystery)
        # Should still produce a schema with string defaults.
        assert "x" in schema["function"]["parameters"]["properties"]

    def test_optional_without_default_is_required_and_nullable(self):
        def f(name: str, note: str | None) -> str:
            """Take an optional but mandatory value."""
            return name

        params = func_to_schema(f)["function"]["parameters"]
        assert params["required"] == ["name", "note"]
        assert params["properties"]["note"] == {
            "anyOf": [{"type": "string"}, {"type": "null"}]
        }

    def test_optional_with_default_is_not_required_and_nullable(self):
        def f(name: str, note: str | None = None) -> str:
            """Take an optional value."""
            return name

        params = func_to_schema(f)["function"]["parameters"]
        assert params["required"] == ["name"]
        assert params["properties"]["note"] == {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
        }

    def test_optional_list_stays_nullable(self):
        def f(items: list[str] | None = None) -> int:
            """Take optional items."""
            return 0

        prop = func_to_schema(f)["function"]["parameters"]["properties"]["items"]
        assert prop["anyOf"] == [
            {"type": "array", "items": {"type": "string"}},
            {"type": "null"},
        ]
        assert prop["default"] is None

    def test_non_optional_unchanged(self):
        def f(count: int) -> int:
            """Plain param."""
            return count

        params = func_to_schema(f)["function"]["parameters"]
        assert params["properties"]["count"] == {"type": "integer"}
        assert params["required"] == ["count"]


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        reg.register(add)
        assert "add" in reg
        assert reg.get("add") is add
        assert len(reg.schemas) == 1

    def test_not_found(self):
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None
        assert "nonexistent" not in reg


# ── Utility tests ────────────────────────────────────────────────

class TestTruncateToolResult:
    def test_short_content_unchanged(self):
        assert truncate_tool_result("short", 100) == "short"

    def test_long_content_truncated(self):
        content = "x" * 200
        result = truncate_tool_result(content, 50)
        assert len(result) < 200
        assert "truncated" in result
        assert "200 chars" in result


class TestMessageHistory:
    def test_init_with_system_prompt(self):
        h = MessageHistory("You are helpful.")
        assert len(h) == 1
        assert h.messages[0].role == "system"

    def test_append_and_reset(self):
        h = MessageHistory("System")
        h.append(Message(role="user", content="Hi"))
        h.append(Message(role="assistant", content="Hello"))
        assert len(h) == 3

        h.reset(keep_system=True)
        assert len(h) == 1
        assert h.messages[0].role == "system"

    def test_reset_no_system(self):
        h = MessageHistory()
        h.append(Message(role="user", content="Hi"))
        h.reset(keep_system=False)
        assert len(h) == 0

    def test_to_api_list(self):
        h = MessageHistory("System")
        h.append(Message(role="user", content="test"))
        api_list = h.to_api_list()
        assert len(api_list) == 2
        assert api_list[0]["role"] == "system"
        assert api_list[1]["content"] == "test"
