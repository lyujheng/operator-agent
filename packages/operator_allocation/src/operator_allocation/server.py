"""MCP bridge server entry point - loads ToolProfile and starts stdio server.

Launched as a subprocess by the main agent:
    python -m operator_allocation.server --profile=identify_server
    python -m operator_allocation.server --profile=process_server --repo_path=/path --sheet_name=FlagGems
    python -m operator_allocation.server --profile=process_server --operators-json=/tmp/ops.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

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
    parser.add_argument(
        "--operators-json",
        default="",
        help="Path to JSON file containing operator data for link_tools state",
    )
    parser.add_argument(
        "--filter-context-json",
        default="",
        help="Path to JSON file containing no_dev filter context (operators, profile, llm_settings)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    profile = PROFILES[args.profile]

    # Initialize in-memory operator state if provided.
    if args.operators_json:
        operators_path = Path(args.operators_json)
        if operators_path.exists():
            try:
                operators = json.loads(operators_path.read_text(encoding="utf-8"))
                from operator_allocation.tools.link_tools import set_operators

                set_operators(operators)
                logger.info(
                    "Loaded %d operators from %s",
                    len(operators),
                    args.operators_json,
                )
            except Exception:
                logger.error(
                    "Failed to load operators from %s",
                    args.operators_json,
                    exc_info=True,
                )

    # Initialize dev table state for data_tools if dev data provided.
    if args.dev_excel_path:
        dev_path = Path(args.dev_excel_path)
        if dev_path.exists():
            try:
                dev_data = json.loads(dev_path.read_text(encoding="utf-8"))
                from operator_allocation.tools.data_tools import set_dev_table

                # dev_data may be a list (single sheet) or a dict (multi-sheet).
                if isinstance(dev_data, list) and args.sheet_name:
                    set_dev_table({args.sheet_name: dev_data})
                elif isinstance(dev_data, dict):
                    set_dev_table(dev_data)
                else:
                    set_dev_table({})
                logger.info("Loaded dev table data from %s", args.dev_excel_path)
            except Exception:
                logger.error(
                    "Failed to load dev table from %s",
                    args.dev_excel_path,
                    exc_info=True,
                )

    # Initialize no_dev filter context if provided.
    if args.filter_context_json:
        ctx_path = Path(args.filter_context_json)
        if ctx_path.exists():
            try:
                ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
                from operator_allocation.tools.no_dev_filter import set_filter_context

                set_filter_context(
                    ctx.get("operators", []),
                    ctx.get("profile", {}),
                    ctx.get("llm_settings", {}),
                )
                logger.info("Loaded filter context from %s", args.filter_context_json)
            except Exception:
                logger.error(
                    "Failed to load filter context from %s",
                    args.filter_context_json,
                    exc_info=True,
                )

    # Build extra context that tools may need.
    extra_context: dict[str, str] = {}
    if args.repo_path:
        extra_context["repo_path"] = args.repo_path
    if args.sheet_name:
        extra_context["sheet_name"] = args.sheet_name
    if args.dev_excel_path:
        # Key must match the tool function parameter name (dev_table_path),
        # so the bridge server auto-injects it into tool calls.
        extra_context["dev_table_path"] = args.dev_excel_path

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
