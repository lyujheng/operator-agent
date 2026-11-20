"""Application entry point and CLI.

Usage:
    # Development
    uv run python -m operator_allocation --config config.toml --input requirement.xlsx

    # Production (after install)
    operator-allocation --config /etc/my-agent/config.toml --input /data/requirement.xlsx
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

from agent_framework.tracing import (
    ConsoleProcessor,
    FileProcessor,
    register_processor,
    trace,
    export_trace_json,
)

from operator_allocation.agents.main_agent import AgentOrchestrator
from operator_allocation.config.settings import AppSettings

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="operator-allocation",
        description="Auto-process operator development requirements",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.toml",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to requirement Excel file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


async def async_main(settings: AppSettings, input_path: str) -> None:
    """Async entry: set up tracing, run orchestrator, export results."""

    # Set up tracing processors.
    output_dir = Path(settings.excel.output_dir)
    log_dir = output_dir / "logs"
    trace_dir = output_dir / "traces"

    register_processor(ConsoleProcessor())
    register_processor(FileProcessor(log_dir))

    with trace("requirement_allocation", input_file=input_path) as t:
        orchestrator = AgentOrchestrator(settings)
        result = await orchestrator.run(input_path)

    # Export full trace.
    trace_path = export_trace_json(t, trace_dir)

    # Print summary.
    stats = result.get("statistics", {})
    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"  总步骤: {stats.get('total_steps', '?')}")
    print(f"  成功: {stats.get('passed', '?')}")
    print(f"  失败: {stats.get('failed', '?')}")
    print(f"  结果文件: {output_dir / 'result.xlsx'}")
    print(f"  Trace 日志: {trace_path}")
    print("=" * 60)


def cli() -> None:
    """CLI entry point (called by ``operator-allocation`` command)."""
    args = parse_args()

    # Configure logging.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load settings.
    try:
        settings = AppSettings.load(args.config)
    except Exception as exc:
        logger.error("Failed to load config '%s': %s", args.config, exc)
        sys.exit(1)

    # Validate input file.
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # Run.
    try:
        asyncio.run(async_main(settings, str(input_path)))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.error("Execution failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
