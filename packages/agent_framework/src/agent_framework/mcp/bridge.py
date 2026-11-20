"""MCPBridgeServer and ToolProfile - dynamic tool loading via profiles.

The bridge server loads Python tool functions specified in a
:class:`ToolProfile` and exposes them over MCP stdio transport,
allowing the framework's MCPStdioServer client to call them.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from importlib import import_module
from typing import Any, Callable

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
)
from pydantic import BaseModel

from agent_framework.agent.tool_registry import func_to_schema

logger = logging.getLogger(__name__)


# ── ToolProfile ──────────────────────────────────────────────────

class ToolProfile(BaseModel):
    """Declarative specification of which tools an MCP server exposes.

    Attributes
    ----------
    name:
        Profile identifier (used as the MCP server name and for CLI lookup).
    description:
        Human-readable purpose of this tool set.
    tool_paths:
        Dotted import paths relative to a base package, e.g.
        ``["repo_tools.list_repo_files", "excel_tools.read_dev_operators"]``.
    """

    name: str
    description: str = ""
    tool_paths: list[str] = []


# ── MCPBridgeServer ──────────────────────────────────────────────

class MCPBridgeServer:
    """MCP server that dynamically loads tools from a :class:`ToolProfile`.

    This bridges the gap between plain Python tool functions defined in a
    business package and the MCP protocol expected by the framework's
    client layer.
    """

    def __init__(
        self,
        profile: ToolProfile,
        tool_base_package: str,
        *,
        extra_context: dict[str, Any] | None = None,
    ) -> None:
        self.profile = profile
        self.tool_base_package = tool_base_package
        self.extra_context = extra_context or {}

        self._functions: dict[str, Callable[..., Any]] = {}
        self._tool_schemas: list[Tool] = []

        self._load_tools()

    # ── tool loading ─────────────────────────────────────────────

    def _load_tools(self) -> None:
        """Import each dotted path and register the function."""
        for tool_path in self.profile.tool_paths:
            try:
                module_name, func_name = tool_path.rsplit(".", 1)
                full_module = f"{self.tool_base_package}.{module_name}"
                module = import_module(full_module)
                func = getattr(module, func_name)

                self._functions[func_name] = func

                # Build MCP Tool schema from the function.
                oai_schema = func_to_schema(func)
                fn_def = oai_schema["function"]
                tool = Tool(
                    name=fn_def["name"],
                    description=fn_def.get("description", ""),
                    input_schema=fn_def.get("parameters", {"type": "object", "properties": {}}),
                )
                self._tool_schemas.append(tool)

                logger.debug("Loaded tool: %s from %s", func_name, full_module)
            except Exception:
                logger.error(
                    "Failed to load tool %s from %s",
                    tool_path,
                    self.tool_base_package,
                    exc_info=True,
                )

    # ── server creation & run ────────────────────────────────────

    def create_low_level_server(self) -> Server:
        """Build a low-level :class:`mcp.server.Server` wired to our tools."""
        functions = self._functions
        schemas = self._tool_schemas
        extra = self.extra_context

        async def handle_list_tools(ctx: Any, params: Any) -> ListToolsResult:
            return ListToolsResult(tools=schemas)

        async def handle_call_tool(ctx: Any, params: CallToolRequestParams) -> CallToolResult:
            name = params.name
            arguments = params.arguments or {}

            func = functions.get(name)
            if func is None:
                available = ", ".join(functions.keys())
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: tool '{name}' not found. Available: {available}")],
                    is_error=True,
                )

            try:
                # Inject extra_context fields that match parameter names.
                sig = inspect.signature(func)
                for key, value in extra.items():
                    if key in sig.parameters and key not in arguments:
                        arguments[key] = value

                if asyncio.iscoroutinefunction(func):
                    result = await func(**arguments)
                else:
                    result = func(**arguments)

                text = json.dumps(result, ensure_ascii=False, default=str)
                return CallToolResult(
                    content=[TextContent(type="text", text=text)],
                    is_error=False,
                )
            except Exception as exc:
                logger.error("Tool %s execution error: %s", name, exc, exc_info=True)
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {exc}")],
                    is_error=True,
                )

        server = Server(
            self.profile.name,
            on_list_tools=handle_list_tools,
            on_call_tool=handle_call_tool,
        )
        return server

    async def run_stdio(self) -> None:
        """Start the MCP bridge server over stdio transport (blocking)."""
        server = self.create_low_level_server()
        logger.info("Starting MCP bridge server '%s' on stdio", self.profile.name)

        async with stdio_server() as (read_stream, write_stream):
            from mcp.server import InitializationOptions

            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name=self.profile.name,
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities=None,
                    ),
                ),
            )


# ── Factory ──────────────────────────────────────────────────────

def create_bridge_server(
    profile: ToolProfile,
    tool_base_package: str,
    *,
    extra_context: dict[str, Any] | None = None,
) -> MCPBridgeServer:
    """Convenience factory for :class:`MCPBridgeServer`."""
    return MCPBridgeServer(
        profile,
        tool_base_package,
        extra_context=extra_context,
    )
