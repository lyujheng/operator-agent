"""LLM response types: Message, ToolCall, LLMResponse.

Pydantic models for the OpenAI-compatible chat completions wire format.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    """Function invocation inside a tool_call."""

    name: str
    arguments: str  # raw JSON string from LLM


class ToolCall(BaseModel):
    """A single tool_call entry from the assistant message."""

    id: str
    type: str = "function"
    function: FunctionCall


class Message(BaseModel):
    """A chat message in OpenAI format.

    Supports all four roles: system, user, assistant, tool.
    """

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # present when role == "tool"

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize to the dict shape expected by the chat completions API."""
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d


class Usage(BaseModel):
    """Token usage reported by the API."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Parsed response from a chat completions call."""

    message: Message
    usage: Usage = Field(default_factory=Usage)
    raw_response: dict[str, Any] | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.message.tool_calls)

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.message.tool_calls or []

    @property
    def content(self) -> str | None:
        return self.message.content
