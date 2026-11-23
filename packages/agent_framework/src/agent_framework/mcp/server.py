"""MCPStdioServer - single MCP server wrapper over stdio transport.

Manages a child process running an MCP server, communicating via
stdin/stdout JSON-RPC through the official ``mcp`` SDK.

Lifecycle is managed as an async context manager::

    async with MCPStdioServer(command="python", args=[...]) as server:
        tools = await server.list_tools()
        result = await server.call_tool("my_tool", {"arg": "value"})
    # server and child process are cleaned up here

The ``__aenter__``/``__aexit__`` pair ensures that the underlying
anyio cancel scopes created by ``stdio_client`` are entered and exited
in the same task, in stack (LIFO) order — a constraint that the
previous ``connect()``/``cleanup()`` split could violate.
"""

from __future__ import annotations

import asyncio
import logging
import random
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult, TextContent, Tool

from agent_framework.exceptions import MCPConnectionError, MCPServerCrashError

logger = logging.getLogger(__name__)

# Default retry parameters for call_tool.
_MAX_RETRIES = 3
_INITIAL_DELAY = 0.5
_MAX_DELAY = 4.0
_MULTIPLIER = 2.0
_JITTER_RATIO = 0.125


class MCPStdioServer:
    """Wraps a single MCP server subprocess.

    Uses ``mcp.client.stdio.stdio_client`` to spawn the child process and
    communicate over stdin/stdout JSON-RPC.

    **Preferred usage** — async context manager::

        async with MCPStdioServer(command="python", args=[...]) as server:
            await server.list_tools()

    This guarantees that the ``AsyncExitStack`` driving ``stdio_client``
    and ``ClientSession`` is entered and exited by the same ``async with``
    block in the same task, satisfying anyio's cancel-scope constraint.

    Parameters
    ----------
    command:
        Executable to launch (e.g. ``"python"``).
    args:
        Command-line arguments (e.g. ``["-m", "my_server"]``).
    env:
        Extra environment variables for the child.
    cwd:
        Working directory for the child process.
    """

    def __init__(
        self,
        *,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd

        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._tools_cache: list[Tool] | None = None

    # ── async context manager (preferred API) ────────────────────

    async def __aenter__(self) -> MCPStdioServer:
        """Start the child process, open the MCP session, and initialise.

        Raises :class:`MCPConnectionError` on any failure and guarantees
        cleanup of partially-acquired resources.
        """
        try:
            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env,
                cwd=self.cwd,
            )

            self._exit_stack = AsyncExitStack()
            await self._exit_stack.__aenter__()

            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )

            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            await self._session.initialize()
            logger.info(
                "MCP server connected: %s %s",
                self.command,
                " ".join(self.args),
            )
            return self
        except Exception as exc:
            # Clean up partially-acquired resources.
            if self._exit_stack is not None:
                await self._exit_stack.__aexit__(type(exc), exc, exc.__traceback__)
                self._exit_stack = None
            self._session = None
            raise MCPConnectionError(
                f"Failed to connect MCP server ({self.command}): {exc}",
                context={
                    "command": self.command,
                    "args": self.args,
                },
            ) from exc

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[type-arg]
        """Shut down the session and terminate the child process."""
        self._session = None
        self._tools_cache = None
        if self._exit_stack is not None:
            try:
                await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            finally:
                self._exit_stack = None

    # ── tool discovery ───────────────────────────────────────────

    async def list_tools(self) -> list[Tool]:
        """Return the tools exposed by this server (cached after first call)."""
        self._ensure_connected()
        if self._tools_cache is None:
            assert self._session is not None
            result = await self._session.list_tools()
            self._tools_cache = list(result.tools)
            logger.debug(
                "Discovered %d tools from %s",
                len(self._tools_cache),
                self.command,
            )
        return self._tools_cache

    # ── tool invocation ──────────────────────────────────────────

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Invoke a tool on the server with exponential-backoff retry.

        Parameters
        ----------
        tool_name:
            The registered tool name.
        arguments:
            JSON-serialisable arguments dict.

        Returns
        -------
        CallToolResult
            The raw MCP result (may contain text, images, etc.).
        """
        self._ensure_connected()
        assert self._session is not None

        delay = _INITIAL_DELAY
        last_exc: Exception | None = None

        for attempt in range(1 + _MAX_RETRIES):
            try:
                result = await self._session.call_tool(tool_name, arguments)
                return result  # type: ignore[return-value]
            except Exception as exc:
                last_exc = exc
                if attempt >= _MAX_RETRIES:
                    break
                jittered = delay * (1 + random.uniform(-_JITTER_RATIO, _JITTER_RATIO))
                logger.warning(
                    "call_tool(%s) attempt %d/%d failed: %s — retrying in %.2fs",
                    tool_name,
                    attempt + 1,
                    _MAX_RETRIES,
                    exc,
                    jittered,
                )
                await asyncio.sleep(jittered)
                delay = min(delay * _MULTIPLIER, _MAX_DELAY)

        # If the session appears dead, flag it as a crash.
        assert last_exc is not None
        raise MCPServerCrashError(
            f"call_tool({tool_name}) failed after {_MAX_RETRIES} retries: {last_exc}",
            context={"tool_name": tool_name, "command": self.command},
        ) from last_exc

    # ── helpers ──────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if self._session is None:
            raise MCPConnectionError(
                "MCPStdioServer is not connected — use 'async with' to manage lifecycle",
                context={"command": self.command},
            )
