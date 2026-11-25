"""Tool configuration for operator allocation MCP servers.

Each profile declares the tools one type of MCP server exposes.
To add a new tool:
  1. Implement the function in the appropriate tools/ module.
  2. Add its dotted path to the matching profile below.
  3. The framework's MCPBridgeServer auto-discovers and registers it on startup.

Path format: "<module_name>.<function_name>"
  - module_name: filename under operator_allocation/tools/ (without .py)
  - function_name: the callable to register as an MCP tool
"""

from agent_framework.mcp.bridge import ToolProfile

# ── Server Profiles ──────────────────────────────────────────────

IDENTIFY_SERVER = ToolProfile(
    name="identify_server",
    description="Operator identification: parse references, search repos, update fields",
    tool_paths=[
        "identify_tools.parse_reference_url",
        "repo_tools.list_repo_files",
        "repo_tools.search_file_content",
        "repo_tools.read_file",
        "link_tools.update_operator_field",
    ],
)

PROCESS_SERVER = ToolProfile(
    name="process_server",
    description="Operator processing: match/add dev operators, link status (per-repo template)",
    tool_paths=[
        "data_tools.read_dev_operators",
        "data_tools.add_dev_operator",
        "repo_tools.list_repo_files",
        "repo_tools.search_file_content",
        "repo_tools.read_file",
        "link_tools.update_link_status",
    ],
)

RESULT_SERVER = ToolProfile(
    name="result_server",
    description="Result consolidation and validation (in-memory)",
    tool_paths=[
        "data_tools.format_result",
        "data_tools.check_result",
    ],
)

NO_DEV_SERVER = ToolProfile(
    name="no_dev_filter_server",
    description="No-dev filtering: classify operators that don't need development",
    tool_paths=[
        "no_dev_filter.filter_no_dev_operators",
        "no_dev_filter.load_no_dev_known_list",
        "no_dev_filter.append_to_pending_list",
    ],
)

# Registry: profile name → ToolProfile (for CLI --profile lookup)
PROFILES: dict[str, ToolProfile] = {
    p.name: p for p in [IDENTIFY_SERVER, PROCESS_SERVER, RESULT_SERVER, NO_DEV_SERVER]
}
