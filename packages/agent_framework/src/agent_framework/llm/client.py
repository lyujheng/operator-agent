"""LLMClient - async HTTP client for OpenAI-compatible chat completions API."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
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
