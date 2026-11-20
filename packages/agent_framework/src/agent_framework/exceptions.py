"""Exception hierarchy for the agent framework.

All custom exceptions inherit from AgentBaseError, which carries
a ``context`` dict for structured error metadata (agent name,
current turn, operator being processed, etc.).
"""

from __future__ import annotations

from typing import Any


class AgentBaseError(Exception):
    """Base class for all agent framework exceptions."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = context or {}


# ── Agent lifecycle ──────────────────────────────────────────────

class MaxTurnsExceeded(AgentBaseError):
    """ReAct loop exceeded the configured max_turns safety limit."""


class MaxRetriesExceeded(AgentBaseError):
    """Reflexion retry count exceeded the configured upper bound."""


# ── LLM errors ───────────────────────────────────────────────────

class LLMError(AgentBaseError):
    """Base class for LLM call-related errors."""


class LLMTimeoutError(LLMError):
    """LLM HTTP request timed out."""


class LLMConnectionError(LLMError):
    """Network-level failure reaching the LLM endpoint."""


class LLMResponseError(LLMError):
    """LLM returned an unparseable or structurally invalid response."""


# ── Tool errors ──────────────────────────────────────────────────

class ToolError(AgentBaseError):
    """Base class for tool execution errors."""


class ToolNotFoundError(ToolError):
    """Attempted to invoke a tool that is not registered."""


class ToolExecutionError(ToolError):
    """A registered tool raised an exception during execution."""


class ToolTimeoutError(ToolError):
    """A tool call exceeded its execution timeout."""


# ── MCP errors ───────────────────────────────────────────────────

class MCPError(AgentBaseError):
    """Base class for MCP communication errors."""


class MCPConnectionError(MCPError):
    """Failed to connect to an MCP server process."""


class MCPServerCrashError(MCPError):
    """An MCP server process exited unexpectedly."""


# ── Validation ───────────────────────────────────────────────────

class ValidationError(AgentBaseError):
    """Data validation failure (illegal field values, format errors)."""
