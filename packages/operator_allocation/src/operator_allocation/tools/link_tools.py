"""Link tools: update_operator_field, update_link_status.

In-memory operations on the operator data structures.  These are
exposed as MCP tools so agents can update operator records during
their ReAct sessions.
"""

from __future__ import annotations

from typing import Any


# Module-level state: the framework injects the working operator list
# via ``set_operators`` before starting agent sessions.
_operators: list[dict[str, Any]] = []


def set_operators(operators: list[dict[str, Any]]) -> None:
    """Inject the operator list into the module for tool access."""
    global _operators
    _operators = operators


def get_operators() -> list[dict[str, Any]]:
    """Return the current operator list."""
    return _operators


def update_operator_field(
    operator_names: list[str],
    field_name: str,
    value: Any,
) -> list[dict[str, Any]]:
    """Batch update a field for multiple operators in memory.

    Args:
        operator_names: List of operator_name values to update.
        field_name: Field to set (e.g. "operator_lib", "link_status").
        value: Value to assign.

    Returns:
        The list of updated operator dicts.
    """
    name_set = set(operator_names)
    updated: list[dict[str, Any]] = []

    for op in _operators:
        if op.get("operator_name") in name_set:
            op[field_name] = value
            updated.append(op)

    return updated


def update_link_status(
    operator_name: str,
    dev_operator_id: int | None,
) -> dict[str, Any]:
    """Set link_status for a single requirement operator.

    Args:
        operator_name: The operator to update.
        dev_operator_id: The dev table operator id, or None to clear.

    Returns:
        {"operator_name": str, "link_status": int | None, "updated": bool}
    """
    for op in _operators:
        if op.get("operator_name") == operator_name:
            op["link_status"] = dev_operator_id
            return {
                "operator_name": operator_name,
                "link_status": dev_operator_id,
                "updated": True,
            }

    return {
        "operator_name": operator_name,
        "link_status": dev_operator_id,
        "updated": False,
    }
