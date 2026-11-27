"""Framework unit tests - MCP server."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp.types import CallToolRequestParams

from agent_framework.mcp.server import MCPStdioServer
from agent_framework.mcp.manager import MCPServerManager
from agent_framework.mcp.converter import mcp_tool_to_function_schema
from agent_framework.mcp.bridge import ToolProfile
from agent_framework.exceptions import MCPConnectionError


# ── Helpers ──────────────────────────────────────────────────────

def _make_mock_stdio_and_session():
    """Create mock stdio_client and ClientSession context managers."""
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()

    mock_stdio_ctx = AsyncMock()
    mock_stdio_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    mock_stdio_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client_ctx = AsyncMock()
    mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_client_ctx.__aexit__ = AsyncMock(return_value=False)

    return mock_stdio_ctx, mock_client_ctx, mock_session


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


class TestMCPBridgeServerCallTool:
    """Tool call errors must come back as MCP errors, not raised exceptions."""

    def _make_bridge(self, func, schemas=None):
        from agent_framework.mcp.bridge import MCPBridgeServer

        class _Stub(MCPBridgeServer):
            def __init__(self):
                self.profile = ToolProfile(name="stub", tool_paths=[])
                self.tool_base_package = "stub"
                self.extra_context = {}
                self._functions = {func.__name__: func}
                self._tool_schemas = schemas or []

        return _Stub()

    def _call(self, bridge, name, arguments):
        server = bridge.create_low_level_server()
        handler = server._request_handlers["tools/call"].handler
        params = CallToolRequestParams(name=name, arguments=arguments)
        return asyncio.run(handler(None, params))

    def test_missing_required_argument_returns_error_result(self):
        def update(operator_name: str, dev_operator_name: str | None = None):
            """Update."""
            return {"ok": True}

        bridge = self._make_bridge(update)
        result = self._call(bridge, "update", {})

        assert result.is_error is True
        assert "update" in result.content[0].text
        assert "operator_name" in result.content[0].text

    def test_omitted_optional_argument_succeeds(self):
        def update(operator_name: str, dev_operator_name: str | None = None):
            """Update."""
            return {"name": operator_name, "link": dev_operator_name}

        bridge = self._make_bridge(update)
        result = self._call(bridge, "update", {"operator_name": "mul"})

        assert not result.is_error
        assert json.loads(result.content[0].text) == {"name": "mul", "link": None}

    def test_runtime_error_returns_error_result(self):
        def boom(x: str):
            """Explode."""
            raise RuntimeError("kaboom")

        bridge = self._make_bridge(boom)
        result = self._call(bridge, "boom", {"x": "1"})

        assert result.is_error is True
        assert "kaboom" in result.content[0].text

    def test_unknown_tool_still_raises(self):
        def update(operator_name: str):
            """Update."""
            return {}

        bridge = self._make_bridge(update)
        with pytest.raises(Exception):
            self._call(bridge, "nope", {})


class TestMCPStdioServerInit:
    def test_construction(self):
        server = MCPStdioServer(
            command="python",
            args=["-m", "operator_allocation.server", "--profile=identify_server"],
        )
        assert server.command == "python"
        assert "--profile=identify_server" in server.args


class TestMCPStdioServerLifecycle:
    """Tests for the async context manager lifecycle (Phase 12)."""

    def test_enter_exit_no_runtime_error(self):
        """async with MCPStdioServer(...) should not raise RuntimeError
        when the underlying stdio_client / ClientSession are properly
        entered and exited in the same task."""

        mock_stdio_ctx, mock_client_ctx, mock_session = _make_mock_stdio_and_session()

        async def _run():
            with patch(
                "agent_framework.mcp.server.stdio_client",
                return_value=mock_stdio_ctx,
            ), patch(
                "agent_framework.mcp.server.ClientSession",
                return_value=mock_client_ctx,
            ):
                server = MCPStdioServer(command="python", args=["-m", "test"])
                async with server as s:
                    assert s is server
                    assert s._session is mock_session
                    mock_session.initialize.assert_awaited_once()

                # After exiting, session and stack should be cleaned up.
                assert server._session is None
                assert server._exit_stack is None

            mock_client_ctx.__aexit__.assert_awaited_once()
            mock_stdio_ctx.__aexit__.assert_awaited_once()

        asyncio.run(_run())

    def test_exit_cleans_up_on_exception(self):
        """Resources should be cleaned up even if an exception occurs
        inside the async with block."""

        mock_stdio_ctx, mock_client_ctx, mock_session = _make_mock_stdio_and_session()

        async def _run():
            with patch(
                "agent_framework.mcp.server.stdio_client",
                return_value=mock_stdio_ctx,
            ), patch(
                "agent_framework.mcp.server.ClientSession",
                return_value=mock_client_ctx,
            ):
                server = MCPStdioServer(command="python", args=["-m", "test"])

                with pytest.raises(ValueError, match="test error"):
                    async with server:
                        raise ValueError("test error")

                assert server._session is None
                assert server._exit_stack is None

            mock_client_ctx.__aexit__.assert_awaited_once()
            mock_stdio_ctx.__aexit__.assert_awaited_once()

        asyncio.run(_run())

    def test_connect_failure_cleans_up(self):
        """If initialize() fails during __aenter__, partial resources
        should still be cleaned up."""

        mock_stdio_ctx, mock_client_ctx, mock_session = _make_mock_stdio_and_session()
        mock_session.initialize = AsyncMock(side_effect=RuntimeError("init failed"))

        async def _run():
            with patch(
                "agent_framework.mcp.server.stdio_client",
                return_value=mock_stdio_ctx,
            ), patch(
                "agent_framework.mcp.server.ClientSession",
                return_value=mock_client_ctx,
            ):
                server = MCPStdioServer(command="python", args=["-m", "test"])

                with pytest.raises(MCPConnectionError, match="init failed"):
                    async with server:
                        pass  # pragma: no cover — never reached

                assert server._session is None
                assert server._exit_stack is None

        asyncio.run(_run())


class TestMCPServerManagerInit:
    def test_empty_manager(self):
        manager = MCPServerManager([])
        assert len(manager._servers) == 0
