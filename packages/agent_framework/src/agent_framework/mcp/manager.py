"""MCPServerManager - multi-server lifecycle manager.

Use as an async context manager to connect several :class:`MCPStdioServer`
instances in parallel and ensure orderly (reverse) cleanup on exit.
"""

from __future__ import annotations

import asyncio
import logging
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

    async def __aenter__(self) -> MCPServerManager:
        await self.connect_all()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.cleanup_all()

    # ── connect ──────────────────────────────────────────────────

    async def connect_all(self) -> None:
        """Connect all servers in parallel.

        Servers that fail to connect are tracked in :attr:`failed_servers`.
        """
        results = await asyncio.gather(
            *(self._safe_connect(srv) for srv in self._servers),
            return_exceptions=False,  # errors handled inside _safe_connect
        )
        # results are already categorised by _safe_connect

    async def _safe_connect(self, server: MCPStdioServer) -> None:
        try:
            await server.connect()
            self.active_servers.append(server)
        except Exception as exc:
            logger.error(
                "Failed to connect MCP server %s: %s",
                server.command,
                exc,
            )
            self.failed_servers.append((server, exc))

    # ── cleanup ──────────────────────────────────────────────────

    async def cleanup_all(self) -> None:
        """Cleanup all servers in reverse order.

        Errors during cleanup are logged but never propagated.
        """
        for server in reversed(self.active_servers):
            try:
                await server.cleanup()
            except Exception:
                logger.debug(
                    "Error cleaning up MCP server %s",
                    server.command,
                    exc_info=True,
                )
        self.active_servers.clear()

        # Also clean up any that failed during connect (partial state).
        for server, _ in reversed(self.failed_servers):
            try:
                await server.cleanup()
            except Exception:
                logger.debug(
                    "Error cleaning up failed MCP server %s",
                    server.command,
                    exc_info=True,
                )
        self.failed_servers.clear()
