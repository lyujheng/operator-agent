"""Framework unit tests - MCP server."""

from __future__ import annotations

import pytest

from agent_framework.mcp.server import MCPStdioServer
from agent_framework.mcp.manager import MCPServerManager
from agent_framework.mcp.converter import mcp_tool_to_function_schema
from agent_framework.mcp.bridge import ToolProfile


class TestToolProfile:
    def test_creation(self):
        tp = ToolProfile(
            name="test_server",
            description="Test tools",
            tool_paths=["module.func_a", "module.func_b"],
        )
        assert tp.name == "test_server"
        assert len(tp.tool_paths) == 2


class TestMCPToolToFunctionSchema:
    def test_basic_conversion(self):
        # Simulate a minimal MCP tool object matching the SDK's Tool model.
        class MockTool:
            name = "search_file_content"
            description = "Search for keyword in files"
            input_schema = {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "Repo root"},
                    "keyword": {"type": "string", "description": "Search term"},
                },
                "required": ["repo_path", "keyword"],
            }

        schema = mcp_tool_to_function_schema(MockTool())
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "search_file_content"
        assert fn["description"] == "Search for keyword in files"
        assert "repo_path" in fn["parameters"]["properties"]
        assert "keyword" in fn["parameters"]["required"]


class TestMCPStdioServerInit:
    def test_construction(self):
        server = MCPStdioServer(
            command="python",
            args=["-m", "operator_allocation.server", "--profile=identify_server"],
        )
        assert server.command == "python"
        assert "--profile=identify_server" in server.args


class TestMCPServerManagerInit:
    def test_empty_manager(self):
        manager = MCPServerManager([])
        assert len(manager._servers) == 0
