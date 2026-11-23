"""Application entry point and CLI.

Usage:
    # Development
    uv run python -m operator_allocation --config config.toml --input requirement.xlsx --dev-table dev_table.xlsx

    # Production (after install)
    operator-allocation --config /etc/my-agent/config.toml --input /data/requirement.xlsx --dev-table /data/dev_table.xlsx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

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
        "--dev-table",
        default="",
        help="Path to dev table Excel file (overrides config)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


# ── Excel → Structured Data Conversion ──────────────────────────

def load_requirement_excel(path: str) -> list[dict[str, Any]]:
    """Load requirement operators from Excel to structured data.

    Returns a list of dicts with keys: operator_name, priority,
    reference_impl, operator_lib (None), link_status (None).
    """
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    if ws is None:
        wb.close()
        raise ValueError(f"No active sheet in {path}")

    rows = list(ws.iter_rows(min_row=2, values_only=True))
    wb.close()

    operators: list[dict[str, Any]] = []
    for row in rows:
        if not row or not row[0]:
            continue
        operators.append({
            "operator_name": str(row[0]).strip(),
            "priority": str(row[1] or "").strip(),
            "reference_impl": str(row[2] or "").strip(),
            "operator_lib": None,
            "link_status": None,
        })
    return operators


def load_dev_table_excel(path: str) -> dict[str, list[dict[str, Any]]]:
    """Load dev table from Excel to structured data.

    Returns a dict keyed by sheet name, each containing a list of
    operator dicts with keys: operator_name, status.
    """
    from openpyxl import load_workbook

    p = Path(path)
    if not p.exists():
        logger.warning("Dev table not found: %s, starting empty", path)
        return {}

    wb = load_workbook(path, read_only=True, data_only=True)
    dev_table: dict[str, list[dict[str, Any]]] = {}

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        operators: list[dict[str, Any]] = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row or row[0] is None:
                continue
            # Excel columns: id(0) | operator_name(1) | status(2)
            operators.append({
                "operator_name": str(row[1] if len(row) > 1 else "").strip(),
                "status": str(row[2] if len(row) > 2 else "not_developed").strip(),
            })
        dev_table[sheet_name] = operators

    wb.close()
    return dev_table


def write_result_excel(
    result: dict[str, Any],
    output_path: str,
) -> str:
    """Write result data back to Excel format.

    Takes the orchestrator's result dict and produces a multi-sheet
    Excel file grouped by operator_lib.
    """
    from openpyxl import Workbook

    from operator_allocation.tools.excel_tools import VALID_LIBS

    wb = Workbook()
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]

    req_result = result.get("requirement_result", [])

    # Group by operator_lib.
    groups: dict[str, list[dict[str, Any]]] = {lib: [] for lib in VALID_LIBS}
    for op in req_result:
        lib = op.get("operator_lib", "unrecognized")
        if lib not in VALID_LIBS:
            lib = "unrecognized"
        groups[lib].append(op)

    header = ["operator_name", "priority", "reference_impl", "operator_lib", "link_status"]
    for lib_name in ["FlagGems", "FlagGems-vllm", "FlagGems-sglang", "unrecognized"]:
        ws = wb.create_sheet(lib_name)
        ws.append(header)
        for op in groups[lib_name]:
            ws.append([
                op.get("operator_name", ""),
                op.get("priority", ""),
                op.get("reference_impl", ""),
                op.get("operator_lib", lib_name),
                op.get("link_status"),
            ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    wb.close()

    logger.info("Result Excel written to %s (%d operators)", output_path, len(req_result))
    return output_path


# ── Async Entry ─────────────────────────────────────────────────

async def async_main(
    settings: AppSettings,
    requirement_operators: list[dict[str, Any]],
    dev_table: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Async entry: set up tracing, run orchestrator, export results."""

    # Set up tracing processors.
    output_dir = Path(settings.excel.output_dir)
    log_dir = output_dir / "logs"
    trace_dir = output_dir / "traces"

    register_processor(ConsoleProcessor())
    register_processor(FileProcessor(log_dir))

    with trace("requirement_allocation", operator_count=len(requirement_operators)) as t:
        orchestrator = AgentOrchestrator(settings)
        result = await orchestrator.run(requirement_operators, dev_table)

    # Export full trace.
    trace_path = export_trace_json(t, trace_dir)

    # Write result Excel from in-memory data.
    result_excel_path = str(output_dir / "result.xlsx")
    write_result_excel(result, result_excel_path)

    # Print summary.
    stats = result.get("statistics", {})
    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"  总算子数: {stats.get('total', '?')}")
    print(f"  已识别: {stats.get('identified', '?')}")
    print(f"  已关联: {stats.get('linked', '?')}")
    print(f"  新增开发算子: {stats.get('new_dev_added', '?')}")
    print(f"  未识别: {stats.get('unrecognized', '?')}")
    if stats.get("warnings"):
        print(f"  警告: {len(stats['warnings'])} 条")
    if stats.get("unresolved_errors"):
        print(f"  未解决错误: {len(stats['unresolved_errors'])} 条")
    print(f"  结果文件: {result_excel_path}")
    print(f"  Trace 日志: {trace_path}")
    print("=" * 60)

    return result


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

    # Determine dev table path.
    dev_table_path = args.dev_table or settings.excel.dev_table_path

    # Convert Excel → structured data at CLI boundary.
    try:
        requirement_operators = load_requirement_excel(str(input_path))
        logger.info("Loaded %d requirement operators", len(requirement_operators))
    except Exception as exc:
        logger.error("Failed to load requirement Excel: %s", exc)
        sys.exit(1)

    try:
        dev_table = load_dev_table_excel(dev_table_path)
        logger.info("Loaded dev table with sheets: %s", list(dev_table.keys()))
    except Exception as exc:
        logger.error("Failed to load dev table Excel: %s", exc)
        sys.exit(1)

    # Run.
    try:
        asyncio.run(async_main(settings, requirement_operators, dev_table))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.error("Execution failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
