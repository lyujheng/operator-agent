"""Framework unit tests - LLM client."""

from __future__ import annotations

import json
import pytest
import httpx

from agent_framework.llm.client import LLMClient, _extract_json_text
from agent_framework.llm.types import LLMResponse, Message, ToolCall
from agent_framework.exceptions import LLMResponseError, LLMTimeoutError


# ── Fixtures ─────────────────────────────────────────────────────

def _mock_completion_response(content: str = "Hello", tool_calls: list | None = None) -> dict:
    """Build a mock chat completions JSON response."""
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
        msg["content"] = None
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


# ── Tests ────────────────────────────────────────────────────────

class TestLLMResponseParsing:
    """Test the static _parse_response method."""

    def test_parse_text_response(self):
        raw = _mock_completion_response("This is a test")
        resp = LLMClient._parse_response(raw)
        assert isinstance(resp, LLMResponse)
        assert resp.content == "This is a test"
        assert not resp.has_tool_calls
        assert resp.usage.total_tokens == 15

    def test_parse_tool_call_response(self):
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "search_file_content",
                    "arguments": '{"repo_path": "/data/repos/vllm", "keyword": "flash_attention"}',
                },
            }
        ]
        raw = _mock_completion_response(tool_calls=tool_calls)
        resp = LLMClient._parse_response(raw)
        assert resp.has_tool_calls
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.function.name == "search_file_content"
        assert "flash_attention" in tc.function.arguments

    def test_parse_empty_choices_raises(self):
        raw = {"choices": []}
        with pytest.raises(LLMResponseError):
            LLMClient._parse_response(raw)

    def test_parse_missing_message_raises(self):
        raw = {"choices": [{"index": 0}]}
        with pytest.raises(LLMResponseError):
            LLMClient._parse_response(raw)


class TestMessageSerialization:
    """Test Message.to_api_dict."""

    def test_system_message(self):
        m = Message(role="system", content="You are helpful.")
        d = m.to_api_dict()
        assert d == {"role": "system", "content": "You are helpful."}

    def test_tool_message(self):
        m = Message(role="tool", content='{"result": 42}', tool_call_id="call_abc")
        d = m.to_api_dict()
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "call_abc"

    def test_assistant_with_tool_calls(self):
        tc = ToolCall(
            id="call_1",
            function={"name": "test_fn", "arguments": "{}"},
        )
        m = Message(role="assistant", tool_calls=[tc])
        d = m.to_api_dict()
        assert "tool_calls" in d
        assert d["tool_calls"][0]["id"] == "call_1"


class TestLLMClientInit:
    """Test client construction."""

    def test_url_trailing_slash_stripped(self):
        client = LLMClient(
            base_url="https://api.example.com/v1/",
            api_key="test-key",
            model="test-model",
        )
        assert client.base_url == "https://api.example.com/v1"


class TestExtractJsonText:
    """Test _extract_json_text helper for robust LLM output parsing."""

    def test_raw_json_object(self):
        raw = '{"steps": [{"step_name": "operator_identify"}]}'
        assert json.loads(_extract_json_text(raw)) == json.loads(raw)

    def test_raw_json_array(self):
        raw = '[{"step_name": "operator_identify"}]'
        assert json.loads(_extract_json_text(raw)) == json.loads(raw)

    def test_markdown_fence_json(self):
        raw = '```json\n{"steps": [{"step_name": "op"}]}\n```'
        result = json.loads(_extract_json_text(raw))
        assert result == {"steps": [{"step_name": "op"}]}

    def test_markdown_fence_with_trailing_text(self):
        """The bug that caused plan parsing to always fail."""
        raw = '```json\n{"steps": [{"step_name": "op"}]}\n```\n\n以上是执行计划 {详见文档}'
        result = json.loads(_extract_json_text(raw))
        assert result == {"steps": [{"step_name": "op"}]}

    def test_json_embedded_in_prose(self):
        raw = 'Here is the plan:\n{"steps": [{"step_name": "op"}]}\nPlease review.'
        result = json.loads(_extract_json_text(raw))
        assert result == {"steps": [{"step_name": "op"}]}

    def test_nested_braces_in_prose(self):
        """raw_decode correctly handles nested JSON within surrounding text."""
        raw = 'Result: {"a": {"b": 1}, "c": [2, 3]} done.'
        result = json.loads(_extract_json_text(raw))
        assert result == {"a": {"b": 1}, "c": [2, 3]}

    def test_whitespace_stripped(self):
        raw = '  \n  {"key": "value"}  \n  '
        result = json.loads(_extract_json_text(raw))
        assert result == {"key": "value"}

    def test_no_json_returns_original(self):
        raw = "No JSON here at all"
        assert _extract_json_text(raw) == raw.strip()
