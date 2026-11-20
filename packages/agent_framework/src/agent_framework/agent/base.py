"""BaseAgent ABC - abstract base with dependency injection.

Also provides ``MessageHistory`` for conversation tracking and
``truncate_tool_result`` for context-window safety.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from agent_framework.agent.hooks import AgentHooks, HookContext
from agent_framework.agent.retry import RetryPolicy
from agent_framework.agent.tool_registry import ToolRegistry
from agent_framework.llm.client import LLMClient
from agent_framework.llm.types import Message

logger = logging.getLogger(__name__)


# ── Utility ──────────────────────────────────────────────────────

def truncate_tool_result(content: str, max_chars: int = 8000) -> str:
    """Truncate a tool result string to *max_chars*, appending a notice."""
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + f"\n... [truncated, total {len(content)} chars]"


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars for English, ~2 for CJK."""
    return len(text) // 3


# ── MessageHistory ───────────────────────────────────────────────

class MessageHistory:
    """Ordered list of :class:`Message` with serialization helpers."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self._messages: list[Message] = []
        if system_prompt:
            self._messages.append(Message(role="system", content=system_prompt))

    def append(self, message: Message) -> None:
        self._messages.append(message)

    def extend(self, messages: list[Message]) -> None:
        self._messages.extend(messages)

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def to_api_list(self) -> list[dict[str, Any]]:
        return [m.to_api_dict() for m in self._messages]

    def reset(self, keep_system: bool = True) -> None:
        """Clear history, optionally preserving the system prompt."""
        if keep_system and self._messages and self._messages[0].role == "system":
            self._messages = [self._messages[0]]
        else:
            self._messages = []

    def total_chars(self) -> int:
        return sum(len(m.content or "") for m in self._messages)

    def __len__(self) -> int:
        return len(self._messages)


# ── BaseAgent ────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Abstract base for all agent types.

    Dependencies are injected via the constructor so each agent instance
    can bind a different model, tool-set, or retry strategy.
    """

    def __init__(
        self,
        name: str,
        llm_client: LLMClient,
        tools: ToolRegistry,
        *,
        hooks: AgentHooks | None = None,
        retry_policy: RetryPolicy | None = None,
        max_turns: int = 10,
    ) -> None:
        self.name = name
        self.llm_client = llm_client
        self.tools = tools
        self.hooks = hooks or AgentHooks()
        self.retry_policy = retry_policy or RetryPolicy()
        self.max_turns = max_turns

    def _hook_context(self, turn: int = 0, **extra: Any) -> HookContext:
        return HookContext(agent_name=self.name, turn=turn, extra=extra)

    @abstractmethod
    async def run(self, input_data: Any) -> Any:
        """Execute the agent's main loop. Subclasses implement this."""
        ...
