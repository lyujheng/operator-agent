"""Tool configuration - ToolProfile instances for MCP server profiles.

Each profile declares the tools one type of MCP server exposes.
To add a new tool:
  1. Implement the function in the appropriate tools/ module.
  2. Add its dotted path to the matching profile below.
  3. The framework's MCPBridgeServer auto-discovers and registers it on startup.

Path format: "<module_name>.<function_name>"
  - module_name: filename under operator_allocation/tools/ (without .py)
  - function_name: the callable to register as an MCP tool
"""
