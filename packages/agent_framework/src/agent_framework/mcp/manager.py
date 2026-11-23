"""MCPServerManager - multi-server lifecycle manager.

Use as an async context manager to connect several :class:`MCPStdioServer`
instances in parallel and ensure orderly (reverse) cleanup on exit.

Each server is entered via its own ``async with`` protocol, so that
anyio cancel scopes are always exited in the same task and stack order
they were entered — matching the constraint fixed in Phase 12.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Sequence

from agent_framework.mcp.server import MCPStdioServer

logger = logging.getLogger(__name__)


class MCPServerManager:
    """Manage the lifecycle of multiple MCP servers.

    Usage::

        servers = [MCPStdioServer(...), MCPStdioServer(...)]
        async with MCPServerManager(servers) as mgr:
            for srv in mgr.active_servers:
                tools = await srv.list_tools()
    """

    def __init__(self, servers: Sequence[MCPStdioServer]) -> None:
        self._servers = list(servers)
        self.active_servers: list[MCPStdioServer] = []
        self.failed_servers: list[tuple[MCPStdioServer, Exception]] = []
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self) -> MCPServerManager:
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        await self.connect_all()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.active_servers.clear()
        self.failed_servers.clear()
        if self._exit_stack is not None:
            try:
                await self._exit_stack.__aexit__(*exc)
            finally:
                self._exit_stack = None

    # ── connect ──────────────────────────────────────────────────

    async def connect_all(self) -> None:
        """Connect all servers in parallel.

        Each server is entered as an async context manager via the
        shared ``AsyncExitStack``.  Servers that fail to connect are
        tracked in :attr:`failed_servers`.
        """
        assert self._exit_stack is not None, "connect_all requires an active exit stack"
        results = await asyncio.gather(
            *(self._safe_connect(srv) for srv in self._servers),
            return_exceptions=False,  # errors handled inside _safe_connect
        )
        # results are already categorised by _safe_connect

    async def _safe_connect(self, server: MCPStdioServer) -> None:
        assert self._exit_stack is not None
        try:
            await self._exit_stack.enter_async_context(server)
            self.active_servers.append(server)
        except Exception as exc:
            logger.error(
                "Failed to connect MCP server %s: %s",
                server.command,
                exc,
            )
            self.failed_servers.append((server, exc))
