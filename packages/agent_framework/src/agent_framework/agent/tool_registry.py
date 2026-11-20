"""ToolRegistry and func_to_schema - tool management and schema generation.

``func_to_schema`` inspects a Python function's signature and docstring
to produce an OpenAI function-calling JSON Schema dict.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable, get_type_hints

logger = logging.getLogger(__name__)

# Python type → JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(tp: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment."""
    origin = getattr(tp, "__origin__", None)

    # Handle Optional (Union[X, None])
    if origin is type(None):
        return {"type": "string"}

    # Handle list[X]
    if origin is list:
        args = getattr(tp, "__args__", None)
        items = _python_type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": items}

    # Handle dict[K, V]
    if origin is dict:
        return {"type": "object"}

    # Handle Union / Optional
    args = getattr(tp, "__args__", None)
    if args and type(None) in args:
        # Optional[X] — use X's schema
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _python_type_to_json_schema(non_none[0])

    # Primitive types
    if tp in _TYPE_MAP:
        return {"type": _TYPE_MAP[tp]}

    # Fallback
    return {"type": "string"}


def func_to_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Generate an OpenAI function-calling tool schema from *func*.

    Uses the function's name, docstring (first line as description),
    and type hints to produce the schema.  Parameters without type hints
    default to ``"string"``.

    Returns a dict in the ``{"type": "function", "function": {...}}`` shape.
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
    doc = inspect.getdoc(func) or ""
    description = doc.split("\n")[0].strip() if doc else func.__name__

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        prop = _python_type_to_json_schema(hints.get(name, str))
        # Try to extract per-param doc from numpy/google style docstrings
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            properties[name]["default"] = (
                param.default
                if isinstance(param.default, (str, int, float, bool, type(None)))
                else str(param.default)
            )

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class ToolRegistry:
    """Maintains ``name → callable`` mapping alongside their JSON Schemas.

    Supports both local Python functions and MCP-backed tools.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._schemas: list[dict[str, Any]] = []

    def register(self, func: Callable[..., Any], *, schema: dict[str, Any] | None = None) -> None:
        """Register a tool function, auto-generating schema if not provided."""
        s = schema or func_to_schema(func)
        name = s["function"]["name"]
        self._tools[name] = func
        self._schemas.append(s)
        logger.debug("Registered tool: %s", name)

    def get(self, name: str) -> Callable[..., Any] | None:
        return self._tools.get(name)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """Return the list of tool schemas for the LLM tools parameter."""
        return list(self._schemas)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
