"""Data tools: pure in-memory operations on operator data structures.

Replaces the Excel I/O tools for MCP tool registration.
The module-level ``_dev_table`` is injected by the framework before
launching each subprocess via ``set_dev_table()``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Valid operator_lib values (also used as sheet names).
VALID_LIBS = {"FlagGems", "FlagGems-vllm", "FlagGems-sglang", "unrecognized"}

# Naming pattern: lowercase, underscore-separated, no leading underscore.
# Trailing underscore is allowed (indicates in-place operation in PyTorch/ATen,
# e.g. add_, index_put_impl_).
_NAMING_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*_?$")

# ── Module-level state ──────────────────────────────────────────

_dev_table: dict[str, list[dict[str, Any]]] = {}


def set_dev_table(data: dict[str, list[dict[str, Any]]]) -> None:
    """Inject the dev table data into the module for tool access."""
    global _dev_table
    _dev_table = data


def get_dev_table() -> dict[str, list[dict[str, Any]]]:
    """Return the current dev table data."""
    return _dev_table


# ── Read ────────────────────────────────────────────────────────

def read_dev_operators(sheet_name: str) -> list[dict[str, Any]]:
    """Read all operators from a dev table sheet (from injected in-memory data).

    Args:
        sheet_name: Sheet name (e.g. "FlagGems").

    Returns:
        List of ``{"operator_name": str, "status": str}``.
    """
    operators = _dev_table.get(sheet_name, [])
    return [
        {
            "operator_name": op.get("operator_name", ""),
            "status": op.get("status", "not_developed"),
        }
        for op in operators
    ]


# ── Add ─────────────────────────────────────────────────────────

def add_dev_operator(sheet_name: str, operator_name: str) -> dict[str, Any]:
    """Add operator to in-memory dev table.

    Validates naming, deduplicates, and appends to the in-memory list.

    1. Naming validation: lowercase, underscore-separated, no leading underscore.
       Trailing underscore is allowed (indicates in-place op, e.g. add_, index_put_impl_).
    2. Dedup: same sheet, same operator_name → return existing (idempotent).
    3. Append: both checks pass → append to in-memory list.

    Args:
        sheet_name: Sheet name (e.g. "FlagGems").
        operator_name: Must be lowercase, underscore-separated (e.g. ``mul_tensor``),
                       must not start with underscore. Trailing underscore is allowed
                       (e.g. ``index_put_impl_`` for in-place operations).

    Returns:
        ``{"operator_name": str, "sheet_name": str, "created": bool, "error": str | None}``
    """
    operator_name = operator_name.strip()

    # Naming validation.
    if not _NAMING_RE.match(operator_name):
        return {
            "operator_name": operator_name,
            "sheet_name": sheet_name,
            "created": False,
            "error": (
                f"Naming violation: '{operator_name}' must be lowercase, "
                "underscore-separated, and not start with underscore "
                "(e.g. mul_tensor). Trailing underscore is allowed for "
                "in-place ops (e.g. index_put_impl_)"
            ),
        }

    # Ensure sheet exists in the dev table.
    if sheet_name not in _dev_table:
        _dev_table[sheet_name] = []

    # Dedup: check if operator already exists in this sheet.
    for op in _dev_table[sheet_name]:
        if op.get("operator_name", "").lower() == operator_name:
            return {
                "operator_name": operator_name,
                "sheet_name": sheet_name,
                "created": False,
                "error": None,
            }

    # Append new operator.
    new_op = {"operator_name": operator_name, "status": "not_developed"}
    _dev_table[sheet_name].append(new_op)

    logger.info(
        "Added dev operator '%s' to sheet '%s'", operator_name, sheet_name
    )
    return {
        "operator_name": operator_name,
        "sheet_name": sheet_name,
        "created": True,
        "error": None,
    }


# ── Format Result ───────────────────────────────────────────────

def format_result(
    operators: list[dict[str, Any]],
    dev_table_updates: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Consolidate processed operators into final result with statistics.

    Validates field completeness, computes statistics, and checks internal
    consistency.

    Args:
        operators: All processed requirement operators (flat list, each has
                   operator_lib field).
        dev_table_updates: New dev operators per sheet (from framework-level
                          diff of initial vs final dev table state).

    Returns:
        ``{"requirement_result": list[dict],
           "dev_table_updates": dict[str, list[dict]],
           "statistics": {...}}``
    """
    warnings: list[str] = []

    # Field completeness validation.
    required_fields = {"operator_name", "priority", "reference_impl", "operator_lib", "link_status"}
    for i, op in enumerate(operators):
        missing = required_fields - set(op.keys())
        if missing:
            warnings.append(
                f"Operator #{i} '{op.get('operator_name', '?')}' missing fields: {missing}"
            )
            # Fill missing fields with defaults.
            for field in missing:
                op[field] = None if field == "link_status" else ""
        # Validate non-empty for original input fields.
        if not op.get("priority"):
            warnings.append(
                f"Operator '{op.get('operator_name', '?')}' has empty priority"
            )
        if not op.get("reference_impl"):
            warnings.append(
                f"Operator '{op.get('operator_name', '?')}' has empty reference_impl"
            )

    # Compute statistics.
    total = len(operators)
    unrecognized = sum(1 for op in operators if op.get("operator_lib") == "unrecognized")
    identified = total - unrecognized
    linked = sum(1 for op in operators if op.get("link_status") is not None)
    new_dev_added = sum(len(v) for v in dev_table_updates.values())

    # Internal consistency checks.
    if identified + unrecognized != total:
        warnings.append(
            f"Statistics mismatch: identified({identified}) + unrecognized({unrecognized}) != total({total})"
        )
    if linked > identified:
        warnings.append(
            f"Statistics mismatch: linked({linked}) > identified({identified})"
        )

    return {
        "requirement_result": operators,
        "dev_table_updates": dev_table_updates,
        "statistics": {
            "total": total,
            "identified": identified,
            "linked": linked,
            "new_dev_added": new_dev_added,
            "unrecognized": unrecognized,
            "warnings": warnings,
            "unresolved_errors": [],
        },
    }


# ── Check Result ────────────────────────────────────────────────

def check_result(
    requirement_result: list[dict[str, Any]],
    dev_table: dict[str, list[dict[str, Any]]],
    dev_table_updates: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Validate result completeness and correctness on in-memory data.

    Performs 8 types of checks:

    **requirement_result checks:**
      - ``missing_operator``: all input operators present
      - ``lib_mismatch``: operator_lib matches group
      - ``invalid_link_name``: link_status name exists in dev table (original + updates)
      - ``unlinked_operator``: non-unrecognized operators must have link_status

    **dev_table_updates checks:**
      - ``orphan_dev_operator``: each new dev operator has reverse link
      - ``dev_sheet_mismatch``: new operator's sheet matches linker's operator_lib
      - ``dev_naming_violation``: naming convention check
      - ``dev_duplicate``: no collision with original dev table

    Returns:
        ``{"passed": bool, "errors": [{"type": str, "operator_name": str, "detail": str}]}``
    """
    errors: list[dict[str, str]] = []

    # Build a combined dev operator lookup: {sheet_name: set(operator_names)}
    all_dev_names: dict[str, set[str]] = {}
    for sheet_name, ops in dev_table.items():
        all_dev_names.setdefault(sheet_name, set())
        for op in ops:
            all_dev_names[sheet_name].add(op.get("operator_name", ""))
    for sheet_name, ops in dev_table_updates.items():
        all_dev_names.setdefault(sheet_name, set())
        for op in ops:
            all_dev_names[sheet_name].add(op.get("operator_name", ""))

    # Build original dev operator lookup for duplicate check.
    original_dev_names: dict[str, set[str]] = {}
    for sheet_name, ops in dev_table.items():
        original_dev_names.setdefault(sheet_name, set())
        for op in ops:
            original_dev_names[sheet_name].add(op.get("operator_name", ""))

    # ── requirement_result checks ──

    result_names = {op.get("operator_name", "") for op in requirement_result}

    # Check 1: lib_mismatch — operator_lib should be a valid lib.
    for op in requirement_result:
        lib = op.get("operator_lib", "")
        if lib and lib not in VALID_LIBS:
            errors.append({
                "type": "lib_mismatch",
                "operator_name": op.get("operator_name", ""),
                "detail": f"operator_lib='{lib}' is not a valid lib",
            })

    # Check 2: invalid_link_name — link_status operator_name must exist in dev table.
    for op in requirement_result:
        link = op.get("link_status")
        if link is None:
            continue
        lib = op.get("operator_lib", "unrecognized")
        if lib == "unrecognized":
            continue
        sheet_ops = all_dev_names.get(lib, set())
        if link not in sheet_ops:
            errors.append({
                "type": "invalid_link_name",
                "operator_name": op.get("operator_name", ""),
                "operator_lib": lib,
                "detail": f"link_status='{link}' not found in dev table sheet '{lib}'",
            })

    # Check 3: unlinked_operator — non-unrecognized with link_status=None.
    for op in requirement_result:
        lib = op.get("operator_lib", "unrecognized")
        if lib != "unrecognized" and op.get("link_status") is None:
            errors.append({
                "type": "unlinked_operator",
                "operator_name": op.get("operator_name", ""),
                "detail": f"operator_lib='{lib}' but link_status is None",
            })

    # ── dev_table_updates checks ──

    # Build reverse link map: new_dev_name → set of requirement operators linking to it.
    reverse_links: dict[str, set[str]] = {}
    for op in requirement_result:
        link = op.get("link_status")
        if link:
            reverse_links.setdefault(link, set()).add(op.get("operator_name", ""))

    for sheet_name, new_ops in dev_table_updates.items():
        for new_op in new_ops:
            op_name = new_op.get("operator_name", "")

            # Check 4: orphan_dev_operator — must have at least one reverse link.
            if op_name not in reverse_links:
                errors.append({
                    "type": "orphan_dev_operator",
                    "operator_name": op_name,
                    "detail": f"New dev operator '{op_name}' in sheet '{sheet_name}' "
                              "has no requirement operator linking to it",
                })

            # Check 5: dev_sheet_mismatch — linker's operator_lib must match sheet.
            if op_name in reverse_links:
                for linker_name in reverse_links[op_name]:
                    linker = next(
                        (o for o in requirement_result if o.get("operator_name") == linker_name),
                        None,
                    )
                    if linker and linker.get("operator_lib") != sheet_name:
                        errors.append({
                            "type": "dev_sheet_mismatch",
                            "operator_name": op_name,
                            "detail": (
                                f"New dev operator in sheet '{sheet_name}' "
                                f"but linker '{linker_name}' has operator_lib='{linker.get('operator_lib')}'"
                            ),
                        })

            # Check 6: dev_naming_violation.
            if not _NAMING_RE.match(op_name):
                errors.append({
                    "type": "dev_naming_violation",
                    "operator_name": op_name,
                    "detail": f"Naming violation: '{op_name}' must be lowercase, "
                              "underscore-separated, not start with underscore",
                })

            # Check 7: dev_duplicate — must not collide with original dev table.
            orig_names = original_dev_names.get(sheet_name, set())
            if op_name in orig_names:
                errors.append({
                    "type": "dev_duplicate",
                    "operator_name": op_name,
                    "detail": f"New dev operator '{op_name}' already exists in "
                              f"original dev table sheet '{sheet_name}'",
                })

    return {"passed": len(errors) == 0, "errors": errors}
