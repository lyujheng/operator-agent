"""Operator pydantic models: RequirementOperator, DevOperator, ResultOperator."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RequirementOperator(BaseModel):
    """A single operator from the requirement Excel input."""

    operator_name: str
    priority: str = ""  # P0, P1, P2
    reference_impl: str = ""  # URL or file path
    operator_lib: str | None = None  # FlagGems / FlagGems-vllm / FlagGems-sglang / unrecognized

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class DevOperator(BaseModel):
    """A single operator from the development table."""

    id: int
    operator_name: str
    status: str = "not_developed"  # not_developed / in_progress / completed


class ResultOperator(BaseModel):
    """An operator record for the result Excel output."""

    operator_name: str
    priority: str = ""
    reference_impl: str = ""
    operator_lib: str = ""  # FlagGems / FlagGems-vllm / FlagGems-sglang / unrecognized
    link_status: int | None = None  # dev table operator id, or None

    @classmethod
    def from_requirement(
        cls, req: RequirementOperator, link_status: int | None = None
    ) -> ResultOperator:
        return cls(
            operator_name=req.operator_name,
            priority=req.priority,
            reference_impl=req.reference_impl,
            operator_lib=req.operator_lib or "unrecognized",
            link_status=link_status,
        )
