"""Excel tools: read_dev_operators, add_dev_operator, output_result_excel, check_result_excel.

All Excel I/O goes through openpyxl.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from openpyxl import Workbook, load_workbook

logger = logging.getLogger(__name__)

# Valid operator_lib values (also used as sheet names in result Excel).
VALID_LIBS = {"FlagGems", "FlagGems-vllm", "FlagGems-sglang", "unrecognized"}


# ── Read ─────────────────────────────────────────────────────────

def read_dev_operators(dev_table_path: str, sheet_name: str) -> list[dict[str, Any]]:
    """Read all operators from a dev table sheet.

    Args:
        dev_table_path: Path to the dev table Excel file.
        sheet_name: Sheet name (e.g. "FlagGems").

    Returns:
        List of ``{"id": int, "operator_name": str, "status": str}``.
    """
    path = Path(dev_table_path)
    if not path.exists():
        logger.warning("Dev table not found: %s", dev_table_path)
        return []

    wb = load_workbook(path, read_only=True, data_only=True)
    if sheet_name not in wb.sheetnames:
        wb.close()
        logger.warning("Sheet '%s' not found in %s", sheet_name, dev_table_path)
        return []

    ws = wb[sheet_name]
    rows = list(ws.iter_rows(min_row=2, values_only=True))  # skip header
    wb.close()

    operators: list[dict[str, Any]] = []
    for row in rows:
        if not row or row[0] is None:
            continue
        operators.append(
            {
                "id": int(row[0]),
                "operator_name": str(row[1] or "").strip(),
                "status": str(row[2] or "not_developed").strip(),
            }
        )
    return operators


# ── Add ──────────────────────────────────────────────────────────

def add_dev_operator(
    dev_table_path: str,
    sheet_name: str,
    operator_name: str,
) -> dict[str, Any]:
    """Add operator to dev table. Idempotent: returns existing record if name matches.

    Args:
        dev_table_path: Path to the dev table Excel file.
        sheet_name: Sheet name.
        operator_name: Must be lowercase, must not start with underscore.

    Returns:
        ``{"id": int, "operator_name": str, "created": bool}``
    """
    # Enforce naming rules.
    operator_name = operator_name.lower().strip()
    if operator_name.startswith("_"):
        operator_name = operator_name.lstrip("_")

    path = Path(dev_table_path)

    if path.exists():
        wb = load_workbook(path)
    else:
        wb = Workbook()
        # Remove default sheet if it's the only one.
        if "Sheet" in wb.sheetnames and len(wb.sheetnames) == 1:
            del wb["Sheet"]

    # Create sheet if absent.
    if sheet_name not in wb.sheetnames:
        ws = wb.create_sheet(sheet_name)
        ws.append(["id", "operator_name", "status"])
    else:
        ws = wb[sheet_name]

    # Check for existing operator (idempotent).
    max_id = 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row and row[0] is not None:
            row_id = int(row[0])
            max_id = max(max_id, row_id)
            if str(row[1] or "").strip().lower() == operator_name:
                wb.close()
                return {"id": row_id, "operator_name": operator_name, "created": False}

    # Insert new.
    new_id = max_id + 1
    ws.append([new_id, operator_name, "not_developed"])
    wb.save(path)
    wb.close()

    logger.info("Added dev operator '%s' (id=%d) to %s/%s", operator_name, new_id, dev_table_path, sheet_name)
    return {"id": new_id, "operator_name": operator_name, "created": True}


# ── Output ───────────────────────────────────────────────────────

def output_result_excel(operators: list[dict[str, Any]], output_path: str) -> str:
    """Generate result Excel grouped by operator_lib into multiple sheets.

    Sheets: FlagGems, FlagGems-vllm, FlagGems-sglang, unrecognized.

    Args:
        operators: List of operator dicts with all result fields.
        output_path: Path for the output Excel file.

    Returns:
        The output file path.
    """
    wb = Workbook()
    # Remove default sheet.
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]

    # Group operators by lib.
    groups: dict[str, list[dict[str, Any]]] = {lib: [] for lib in VALID_LIBS}
    for op in operators:
        lib = op.get("operator_lib", "unrecognized")
        if lib not in VALID_LIBS:
            lib = "unrecognized"
        groups[lib].append(op)

    # Write each group to its own sheet.
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

    logger.info("Result Excel written to %s (%d operators)", output_path, len(operators))
    return output_path


# ── Check ────────────────────────────────────────────────────────

def check_result_excel(
    result_path: str,
    requirement_operators: list[dict[str, Any]],
    dev_table_path: str,
) -> dict[str, Any]:
    """Validate result Excel completeness and correctness.

    Checks:
      1. All requirement operators are present in the result.
      2. Each operator's operator_lib matches the sheet it is in.
      3. Each link_status id exists in the corresponding dev table sheet.

    Returns:
        ``{"passed": bool, "errors": [{"type": str, "operator_name": str, "detail": str}]}``
    """
    errors: list[dict[str, str]] = []
    path = Path(result_path)

    if not path.exists():
        return {
            "passed": False,
            "errors": [{"type": "file_missing", "operator_name": "", "detail": f"Result file not found: {result_path}"}],
        }

    wb = load_workbook(path, read_only=True, data_only=True)

    # Collect all operators from result sheets.
    result_ops: dict[str, dict[str, Any]] = {}
    for sheet_name in wb.sheetnames:
        if sheet_name not in VALID_LIBS:
            continue
        ws = wb[sheet_name]
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row or not row[0]:
                continue
            op_name = str(row[0]).strip()
            result_ops[op_name] = {
                "operator_name": op_name,
                "operator_lib": str(row[3] or sheet_name).strip(),
                "link_status": row[4],
                "sheet": sheet_name,
            }
    wb.close()

    # Check 1: All requirement operators present.
    for req in requirement_operators:
        name = req.get("operator_name", "").strip()
        if name and name not in result_ops:
            errors.append({
                "type": "missing_operator",
                "operator_name": name,
                "detail": f"Requirement operator '{name}' not found in result Excel",
            })

    # Check 2: operator_lib matches sheet.
    for op_name, info in result_ops.items():
        if info["operator_lib"] != info["sheet"]:
            errors.append({
                "type": "lib_mismatch",
                "operator_name": op_name,
                "detail": f"operator_lib='{info['operator_lib']}' but in sheet '{info['sheet']}'",
            })

    # Check 3: link_status ids exist in dev table.
    dev_path = Path(dev_table_path)
    if dev_path.exists():
        dev_wb = load_workbook(dev_path, read_only=True, data_only=True)
        dev_ids: dict[str, set[int]] = {}
        for sn in dev_wb.sheetnames:
            ids: set[int] = set()
            for row in dev_wb[sn].iter_rows(min_row=2, values_only=True):
                if row and row[0] is not None:
                    ids.add(int(row[0]))
            dev_ids[sn] = ids
        dev_wb.close()

        for op_name, info in result_ops.items():
            link = info["link_status"]
            if link is None:
                continue
            try:
                link_id = int(link)
            except (ValueError, TypeError):
                errors.append({
                    "type": "invalid_link_id",
                    "operator_name": op_name,
                    "detail": f"link_status '{link}' is not a valid integer",
                })
                continue
            sheet = info["sheet"]
            if sheet in dev_ids and link_id not in dev_ids[sheet]:
                errors.append({
                    "type": "invalid_link_id",
                    "operator_name": op_name,
                    "detail": f"link_status {link_id} not found in dev table sheet '{sheet}'",
                })

    return {"passed": len(errors) == 0, "errors": errors}
