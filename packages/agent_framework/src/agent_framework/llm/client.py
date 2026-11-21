"""LLMClient - async HTTP client for OpenAI-compatible chat completions API."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from pydantic import BaseModel

from agent_framework.exceptions import LLMResponseError
from agent_framework.llm.retry import retry_with_backoff
from agent_framework.llm.types import (
    FunctionCall,
    LLMResponse,
    Message,
    ToolCall,
    Usage,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Async client for OpenAI-compatible ``/chat/completions`` endpoints.

    Parameters
    ----------
    base_url:
        API base URL (e.g. ``https://api.openai.com/v1``).
    api_key:
        Bearer token.
    model:
        Default model name.
    timeout:
        Per-request timeout in seconds (default 120).
    max_retries:
        Maximum retries on transient errors (default 3).
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        max_retries: int = 3,
        structured_output_mode: str = "openai",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.structured_output_mode = structured_output_mode
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    # ── public API ───────────────────────────────────────────────

    async def chat_completions(
        self,
        messages: list[Message] | list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        model: str | None = None,
    ) -> LLMResponse:
        """Send a chat completions request and return a parsed :class:`LLMResponse`.

        Parameters
        ----------
        messages:
            Conversation history (Message objects or raw dicts).
        tools:
            Tool definitions in OpenAI function-calling schema.
        tool_choice:
            ``"auto"`` (default), ``"none"``, or ``"required"``.
        model:
            Override the default model for this single call.
        """
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": [
                m.to_api_dict() if isinstance(m, Message) else m for m in messages
            ],
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        raw = await retry_with_backoff(
            lambda: self._post(payload),
            max_retries=self.max_retries,
        )
        return self._parse_response(raw)

    async def close(self) -> None:
        """Gracefully close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ── internals ────────────────────────────────────────────────

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to /chat/completions and return the JSON body."""
        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_response(raw: dict[str, Any]) -> LLMResponse:
        """Turn the raw JSON dict into an :class:`LLMResponse`."""
        try:
            choice = raw["choices"][0]
            msg_data = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                f"Unexpected response structure: {json.dumps(raw)[:500]}",
                context={"raw_keys": list(raw.keys()) if isinstance(raw, dict) else None},
            ) from exc

        # Parse tool_calls if present.
        tool_calls: list[ToolCall] | None = None
        if msg_data.get("tool_calls"):
            tool_calls = []
            for tc in msg_data["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        type=tc.get("type", "function"),
                        function=FunctionCall(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                )

        message = Message(
            role=msg_data.get("role", "assistant"),
            content=msg_data.get("content"),
            tool_calls=tool_calls,
        )

        usage_data = raw.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return LLMResponse(message=message, usage=usage, raw_response=raw)

    # ── structured output ────────────────────────────────────────

    async def get_structured_output(
        self,
        messages: list[Message] | list[dict[str, Any]],
        schema: type[BaseModel],
    ) -> dict[str, Any] | None:
        """Extract structured output from a completed conversation.

        Dispatches based on ``structured_output_mode``:

        - ``"openai"``: extra LLM call with ``response_format`` json_schema
        - ``"claude"``: extra LLM call with ``output_config.format`` json_schema
        - ``"fallback"``: scan tool-call history, no extra LLM call

        Returns a dict matching *schema*, or None on failure.
        """
        if self.structured_output_mode == "openai":
            return await self._openai_structured(messages, schema)
        elif self.structured_output_mode == "claude":
            return await self._claude_structured(messages, schema)
        else:
            return self._extract_from_history(messages, schema)

    async def _openai_structured(
        self,
        messages: list[Message] | list[dict[str, Any]],
        schema: type[BaseModel],
    ) -> dict[str, Any] | None:
        """OpenAI-style: extra call with response_format json_schema."""
        extraction_msg = {"role": "user", "content": "请将你的处理结果整理为指定的 JSON 格式输出。"}
        api_messages = [
            m.to_api_dict() if isinstance(m, Message) else m for m in messages
        ]
        api_messages.append(extraction_msg)

        json_schema = schema.model_json_schema()
        # Ensure additionalProperties: false for strict mode
        json_schema.setdefault("additionalProperties", False)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": True,
                    "schema": json_schema,
                },
            },
        }
        try:
            raw = await retry_with_backoff(
                lambda: self._post(payload),
                max_retries=self.max_retries,
            )
            content = raw["choices"][0]["message"].get("content", "")
            return json.loads(content)
        except Exception:
            logger.warning("OpenAI structured output extraction failed", exc_info=True)
            return None

    async def _claude_structured(
        self,
        messages: list[Message] | list[dict[str, Any]],
        schema: type[BaseModel],
    ) -> dict[str, Any] | None:
        """Claude-style: extra call with output_config.format json_schema."""
        extraction_msg = {"role": "user", "content": "请将你的处理结果整理为指定的 JSON 格式输出。"}
        api_messages = [
            m.to_api_dict() if isinstance(m, Message) else m for m in messages
        ]
        api_messages.append(extraction_msg)

        json_schema = schema.model_json_schema()
        json_schema.setdefault("additionalProperties", False)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": json_schema,
                },
            },
        }
        try:
            raw = await retry_with_backoff(
                lambda: self._post(payload),
                max_retries=self.max_retries,
            )
            content = raw["choices"][0]["message"].get("content", "")
            return json.loads(content)
        except Exception:
            logger.warning("Claude structured output extraction failed", exc_info=True)
            return None

    @staticmethod
    def _extract_from_history(
        messages: list[Message] | list[dict[str, Any]],
        schema: type[BaseModel],
    ) -> dict[str, Any] | None:
        """Fallback: scan tool-call history and assistant messages for data.

        Collects all key-value pairs from tool results and the last
        assistant message, then validates against *schema* to produce
        a complete result dict.  Fields not found in tool results are
        filled from the assistant's final text (if it looks like JSON)
        or set to schema defaults / None.
        """
        raw_msgs = [m.to_api_dict() if isinstance(m, Message) else m for m in messages]

        # Collect all fields from tool results (reverse scan, first wins).
        collected: dict[str, Any] = {}
        for msg in reversed(raw_msgs):
            if msg.get("role") == "tool":
                try:
                    tool_data = json.loads(msg.get("content", "{}"))
                    if isinstance(tool_data, dict):
                        for k, v in tool_data.items():
                            if k not in collected:
                                collected[k] = v
                except (json.JSONDecodeError, TypeError):
                    continue

        # Try to extract fields from the last assistant text message.
        for msg in reversed(raw_msgs):
            if msg.get("role") == "assistant" and msg.get("content"):
                try:
                    assistant_data = json.loads(msg["content"])
                    if isinstance(assistant_data, dict):
                        for k, v in assistant_data.items():
                            if k not in collected:
                                collected[k] = v
                except (json.JSONDecodeError, TypeError):
                    pass
                break  # Only check the last assistant message.

        # Build result matching schema fields.
        schema_fields = schema.model_fields
        result: dict[str, Any] = {}
        for field_name, field_info in schema_fields.items():
            if field_name in collected:
                result[field_name] = collected[field_name]
            elif field_info.default is not None:
                result[field_name] = field_info.default
            else:
                result[field_name] = None

        # Validate with schema; return None if required fields are missing.
        try:
            validated = schema(**result)
            return validated.model_dump()
        except Exception:
            logger.warning(
                "Fallback extraction could not build a valid %s from history",
                schema.__name__,
            )
            return result  # Return best-effort dict even if validation fails.
