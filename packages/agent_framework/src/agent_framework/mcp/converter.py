"""MCP tool schema conversion utilities.

Bridges the gap between MCP's native tool definitions and OpenAI's
function-calling JSON Schema format used by the LLM layer.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.types import TextContent, Tool

from agent_framework.mcp.server import MCPStdioServer

logger = logging.getLogger(__name__)


def mcp_tool_to_function_schema(tool: Tool) -> dict[str, Any]:
    """Convert an MCP :class:`Tool` to OpenAI function-calling schema.

    Returns a dict of the form::

        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": { ... }
            }
        }
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or tool.name,
            "parameters": tool.input_schema,
        },
    }


async def invoke_mcp_tool(
    server: MCPStdioServer,
    tool_name: str,
    arguments_json: str,
) -> str:
    """Parse a JSON arguments string, call an MCP tool, and return the result as text.

    Parameters
    ----------
    server:
        The connected MCP server to invoke the tool on.
    tool_name:
        Name of the tool to call.
    arguments_json:
        Raw JSON string of the tool arguments (as returned by the LLM).

    Returns
    -------
    str
        The tool result serialised as a plain string.
    """
    try:
        arguments = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as exc:
        return f"Error: invalid JSON arguments — {exc}"

    result = await server.call_tool(tool_name, arguments)

    # Extract text from the CallToolResult content list.
    parts: list[str] = []
    for item in result.content:
        if isinstance(item, TextContent):
            parts.append(item.text)
        else:
            # For non-text content (images, etc.), include a placeholder.
            parts.append(f"[{type(item).__name__}]")

    output = "\n".join(parts) if parts else ""

    if result.is_error:
        return f"Error: {output}"

    return output
