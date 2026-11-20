"""MCP bridge server entry point - loads ToolProfile and starts stdio server.

Launched as a subprocess by the main agent:
    python -m operator_allocation.server --profile=identify_server
    python -m operator_allocation.server --profile=process_server --repo_path=/path --sheet_name=FlagGems
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from agent_framework.mcp.bridge import MCPBridgeServer, create_bridge_server
from operator_allocation.config.tool_config import PROFILES

logger = logging.getLogger(__name__)

TOOL_BASE_PACKAGE = "operator_allocation.tools"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Operator Allocation MCP Server")
    parser.add_argument(
        "--profile",
        required=True,
        choices=list(PROFILES.keys()),
        help="Tool profile to load",
    )
    parser.add_argument("--repo_path", default="", help="Repository path (for process_server)")
    parser.add_argument("--sheet_name", default="", help="Dev table sheet name (for process_server)")
    parser.add_argument("--dev_excel_path", default="", help="Dev table Excel path")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    profile = PROFILES[args.profile]

    # Build extra context that tools may need.
    extra_context: dict[str, str] = {}
    if args.repo_path:
        extra_context["repo_path"] = args.repo_path
    if args.sheet_name:
        extra_context["sheet_name"] = args.sheet_name
    if args.dev_excel_path:
        extra_context["dev_excel_path"] = args.dev_excel_path

    logger.info(
        "Starting MCP server: profile=%s, extra=%s",
        profile.name,
        extra_context,
    )

    server = create_bridge_server(
        profile,
        tool_base_package=TOOL_BASE_PACKAGE,
        extra_context=extra_context,
    )
    await server.run_stdio()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    )
    asyncio.run(main())
