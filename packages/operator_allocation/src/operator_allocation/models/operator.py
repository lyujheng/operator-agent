"""Operator pydantic models: RequirementOperator, DevOperator, ResultOperator.

Also defines structured output schemas (IdentifyResult, ProcessResult)
used by ReActAgent sessions.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Default status for newly added dev operators.
DEFAULT_DEV_STATUS = "not_developed"


class RequirementOperator(BaseModel):
    """A single operator from the requirement Excel input."""

    operator_name: str
    priority: str = ""  # P0, P1, P2
    reference_impl: str = ""  # URL or file path
    operator_lib: str | None = None  # FlagGems / FlagGems-vllm / FlagGems-sglang / unrecognized
    link_status: str | None = None  # dev table operator name, or None
    skip_dev: bool = False  # True if operator does not need development
    skip_reason: str | None = None  # e.g. "exact_match:view_shape", "llm:attribute_query"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class DevOperator(BaseModel):
    """A single operator from the development table."""

    operator_name: str
    status: str = DEFAULT_DEV_STATUS  # not_developed / in_progress / completed


class ResultOperator(BaseModel):
    """An operator record for the result Excel output."""

    operator_name: str
    priority: str = ""
    reference_impl: str = ""
    operator_lib: str = ""  # FlagGems / FlagGems-vllm / FlagGems-sglang / unrecognized
    link_status: str | None = None  # dev table operator name, or None

    @classmethod
    def from_requirement(
        cls, req: RequirementOperator, link_status: str | None = None
    ) -> ResultOperator:
        return cls(
            operator_name=req.operator_name,
            priority=req.priority,
            reference_impl=req.reference_impl,
            operator_lib=req.operator_lib or "unrecognized",
            link_status=link_status,
        )


# ── Structured output schemas for ReAct sessions ────────────────


class IdentifyResult(BaseModel):
    """identify_agent ReAct session structured return value."""

    operator_name: str  # requirement operator being processed
    operator_lib: str  # FlagGems / FlagGems-vllm / FlagGems-sglang / unrecognized
    summary: str  # LLM summary of this identification session


class ProcessResult(BaseModel):
    """process_agent ReAct session structured return value."""

    operator_name: str  # requirement operator being processed
    link_status: str | None  # linked dev operator name, or None
    new_dev_operator: dict | None = None  # {"operator_name": str, "sheet_name": str} or None
    summary: str  # LLM summary of this processing session
