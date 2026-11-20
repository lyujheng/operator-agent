"""AgentHooks and HookContext - lifecycle hook interface.

All methods default to no-ops; override selectively in business code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HookContext:
    """Contextual info passed to every hook callback."""

    agent_name: str = ""
    turn: int = 0
    trace_id: str = ""
    span_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class AgentHooks:
    """Lifecycle hooks for agent loop key-points.

    Override individual methods for approval gates, rate-limiting,
    metrics collection, or UI push.
    """

    async def on_turn_start(self, context: HookContext) -> None:
        pass

    async def on_turn_end(self, context: HookContext) -> None:
        pass

    async def on_tool_start(self, context: HookContext, tool_name: str) -> None:
        pass

    async def on_tool_end(
        self, context: HookContext, tool_name: str, result: str
    ) -> None:
        pass

    async def on_llm_start(self, context: HookContext) -> None:
        pass

    async def on_llm_end(self, context: HookContext, response: Any) -> None:
        pass

    async def on_error(self, context: HookContext, error: Exception) -> None:
        pass
