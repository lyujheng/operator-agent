"""ExecutionPlan, PlanStep, and StepResultSummarizer.

Structured output models for the PlanSolve agent's planning phase,
plus the ABC that business layers implement to control how step
results are summarized into the main_agent's message history.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A single step in the execution plan."""

    step_name: str  # e.g. "operator_identify", "task_dispatch"
    description: str
    agent_type: str  # "identify" | "process" | "result"
    expected_criteria: str  # validation criteria for reflexion check
    depends_on: list[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    """Structured plan that the LLM generates and the framework executes."""

    plan_id: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "pending"  # "pending" | "executing" | "completed" | "failed"
    current_step_index: int = 0  # for resume after interruption


class StepResult(BaseModel):
    """Outcome of executing one plan step."""

    step_name: str
    passed: bool = False
    data: Any = None  # full result (kept in framework memory)
    summary: dict[str, Any] = Field(default_factory=dict)  # goes into messages
    error: str | None = None
    retries: int = 0


class StepResultSummarizer(ABC):
    """Interface for converting a step's full result into a compact summary.

    Business layers implement one per step type and register them
    in a ``summarizer_registry``.
    """

    @abstractmethod
    def summarize(self, step: PlanStep, result: Any) -> dict[str, Any]:
        """Return a JSON-serializable summary dict for the main agent's messages."""
        ...
