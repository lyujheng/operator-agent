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
    operators: list[dict[str, Any]],
    field_name: str,
    value: Any,
) -> list[dict[str, Any]]:
    """Batch update a field for multiple operators in memory.

    Args:
        operators: List of operator dicts to update (each must contain "operator_name").
        field_name: Field to set (e.g. "operator_lib", "link_status").
        value: Value to assign.

    Returns:
        The list of updated operator dicts.
    """
    name_set = {op["operator_name"] for op in operators if "operator_name" in op}
    updated: list[dict[str, Any]] = []

    for op in _operators:
        if op.get("operator_name") in name_set:
            op[field_name] = value
            updated.append(op)

    return updated


def update_link_status(
    operator_name: str,
    dev_operator_name: str | None = None,
) -> dict[str, Any]:
    """Set link_status for a single requirement operator.

    Args:
        operator_name: The requirement operator to update.
        dev_operator_name: The dev table operator name to link, or None to clear.
            Omit it (default None) to clear link_status.

    Returns:
        {"operator_name": str, "link_status": str | None, "updated": bool}
    """
    for op in _operators:
        if op.get("operator_name") == operator_name:
            op["link_status"] = dev_operator_name
            return {
                "operator_name": operator_name,
                "link_status": dev_operator_name,
                "updated": True,
            }

    return {
        "operator_name": operator_name,
        "link_status": dev_operator_name,
        "updated": False,
    }
