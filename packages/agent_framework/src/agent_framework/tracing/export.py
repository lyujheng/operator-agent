"""Trace JSON export utilities."""

from __future__ import annotations

import json
from pathlib import Path

from agent_framework.tracing.trace import Trace


def export_trace_json(trace: Trace, output_dir: str | Path) -> Path:
    """Write the full *trace* (including all spans) to a JSON file.

    Returns the path of the exported file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{trace.trace_id}.json"
    path.write_text(
        trace.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return path
