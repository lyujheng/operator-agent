"""PlanSolveAgent - plan-then-execute loop for main_agent.

Two-phase loop:
  Phase 1: LLM generates an ``ExecutionPlan`` (structured output).
  Phase 2: Steps are executed sequentially with per-step reflexion.
           Each step's result is summarized (by a registered
           ``StepResultSummarizer``) before being appended to messages.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Awaitable

from agent_framework.agent.base import BaseAgent, MessageHistory
from agent_framework.agent.plan_types import (
    ExecutionPlan,
    PlanStep,
    StepResult,
    StepResultSummarizer,
)
from agent_framework.exceptions import MaxRetriesExceeded
from agent_framework.llm.types import Message
from agent_framework.tracing.context import span

logger = logging.getLogger(__name__)


class PlanSolveAgent(BaseAgent):
    """Orchestrating agent that plans first, then executes step by step.

    Parameters
    ----------
    system_prompt:
        System message for the planning + execution phases.
    max_retries:
        Reflexion retry limit per step (default 10).
    step_executors:
        ``step_name → async callable(step, context) → Any`` map.
        Business code registers the actual execution logic per step type.
    summarizers:
        ``step_name → StepResultSummarizer`` map.
    """

    def __init__(
        self,
        *args: Any,
        system_prompt: str = "",
        max_retries: int = 10,
        step_executors: dict[str, Callable[..., Awaitable[Any]]] | None = None,
        summarizers: dict[str, StepResultSummarizer] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.step_executors = step_executors or {}
        self.summarizers = summarizers or {}

    async def run(self, input_data: Any) -> dict[str, Any]:
        """Execute the full plan-and-solve loop.

        Parameters
        ----------
        input_data:
            User input (typically the requirement Excel path).

        Returns
        -------
        A dict with ``plan``, ``results``, and ``statistics``.
        """
        history = MessageHistory(self.system_prompt)
        step_results: dict[str, StepResult] = {}

        # Build user message.
        if isinstance(input_data, str):
            user_content = input_data
        else:
            user_content = json.dumps(input_data, ensure_ascii=False, default=str)
        history.append(Message(role="user", content=user_content))

        # ── Phase 1: Planning ────────────────────────────────────
        plan = await self._generate_plan(history)
        plan.status = "executing"

        # ── Phase 2: Step-by-step execution ──────────────────────
        for idx, step in enumerate(plan.steps):
            plan.current_step_index = idx
            step_result = await self._execute_step_with_retries(
                step, history, step_results
            )
            step_results[step.step_name] = step_result

        plan.status = "completed"

        # ── Statistics ───────────────────────────────────────────
        stats = {
            "total_steps": len(plan.steps),
            "passed": sum(1 for r in step_results.values() if r.passed),
            "failed": sum(1 for r in step_results.values() if not r.passed),
        }

        return {
            "plan": plan.model_dump(),
            "results": {k: v.model_dump() for k, v in step_results.items()},
            "statistics": stats,
        }

    # ── Phase 1 ──────────────────────────────────────────────────

    async def _generate_plan(self, history: MessageHistory) -> ExecutionPlan:
        """Ask the LLM to produce an ExecutionPlan."""
        with span("plan_generation", span_type="llm_call", agent_name=self.name):
            planning_msg = Message(
                role="system",
                content=(
                    "Analyze the input and create a step-by-step execution plan. "
                    "Return a JSON object with a 'steps' array. Each step has: "
                    "step_name, description, agent_type, expected_criteria, depends_on (list of step_names). "
                    "Available step types: operator_identify, task_dispatch, output_result, check_result."
                ),
            )
            history.append(planning_msg)

            response = await self.llm_client.chat_completions(history.messages)
            content = response.content or ""
            history.append(response.message)

        # Try to parse the plan from LLM output.
        plan = self._parse_plan(content)
        if not plan.steps:
            # Use a default plan if LLM didn't produce one.
            plan = self._default_plan()
            logger.warning("LLM plan was empty; using default plan")
        return plan

    def _parse_plan(self, content: str) -> ExecutionPlan:
        """Extract an ExecutionPlan from LLM text (may contain markdown fences)."""
        # Strip markdown code fences if present.
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines.
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
            if "steps" in data:
                return ExecutionPlan(**data)
            # Maybe the JSON is a list of steps directly.
            if isinstance(data, list):
                return ExecutionPlan(steps=[PlanStep(**s) for s in data])
        except (json.JSONDecodeError, Exception) as exc:
            logger.debug("Failed to parse plan from LLM: %s", exc)

        return ExecutionPlan()

    @staticmethod
    def _default_plan() -> ExecutionPlan:
        """Fallback plan matching the requirement doc flow."""
        return ExecutionPlan(
            steps=[
                PlanStep(
                    step_name="operator_identify",
                    description="识别需求表中各算子所属的 operator_lib",
                    agent_type="identify",
                    expected_criteria=(
                        "所有算子的 operator_lib 字段必须为 "
                        "FlagGems / FlagGems-vllm / FlagGems-sglang / unrecognized 之一"
                    ),
                ),
                PlanStep(
                    step_name="task_dispatch",
                    description="将已识别的算子分发到对应仓库的子 agent 进行关联",
                    agent_type="process",
                    expected_criteria=(
                        "所有算子的 link_status 字段为数字 id 或为空(unrecognized 算子)"
                    ),
                    depends_on=["operator_identify"],
                ),
                PlanStep(
                    step_name="output_result",
                    description="生成按 operator_lib 分组的结果 Excel",
                    agent_type="result",
                    expected_criteria="result_excel 文件已生成且包含所有算子",
                    depends_on=["task_dispatch"],
                ),
                PlanStep(
                    step_name="check_result",
                    description="校验结果完整性和正确性",
                    agent_type="result",
                    expected_criteria="所有校验通过: 算子完整、lib 匹配 sheet、link_id 有效",
                    depends_on=["output_result"],
                ),
            ]
        )

    # ── Phase 2 ──────────────────────────────────────────────────

    async def _execute_step_with_retries(
        self,
        step: PlanStep,
        history: MessageHistory,
        prior_results: dict[str, StepResult],
    ) -> StepResult:
        """Execute a step with reflexion-based retries."""
        result = StepResult(step_name=step.step_name)

        for attempt in range(1, self.max_retries + 1):
            result.retries = attempt

            with span(
                f"step_{step.step_name}",
                span_type="step",
                agent_name=self.name,
                attempt=attempt,
            ) as s:
                try:
                    # Dispatch to the registered executor.
                    executor = self.step_executors.get(step.step_name)
                    if executor is None:
                        result.error = f"No executor registered for step '{step.step_name}'"
                        result.passed = False
                        logger.error(result.error)
                        break

                    raw_result = await executor(step, prior_results)
                    result.data = raw_result

                    # Summarize and append to history.
                    summarizer = self.summarizers.get(step.step_name)
                    if summarizer:
                        result.summary = summarizer.summarize(step, raw_result)
                    else:
                        result.summary = {"step": step.step_name, "data": str(raw_result)[:500]}

                    history.append(
                        Message(
                            role="tool",
                            content=json.dumps(result.summary, ensure_ascii=False, default=str),
                        )
                    )

                    # Reflexion: validate result.
                    passed = await self._validate_step(step, result, history)
                    result.passed = passed

                    if passed:
                        s.output_data = result.summary
                        logger.info(
                            "[%s] Step '%s' passed on attempt %d",
                            self.name,
                            step.step_name,
                            attempt,
                        )
                        return result
                    else:
                        logger.warning(
                            "[%s] Step '%s' retry %d/%d: criteria not met",
                            self.name,
                            step.step_name,
                            attempt,
                            self.max_retries,
                        )

                except Exception as exc:
                    result.error = str(exc)
                    s.fail(str(exc))
                    logger.error(
                        "[%s] Step '%s' attempt %d error: %s",
                        self.name,
                        step.step_name,
                        attempt,
                        exc,
                    )

        # All retries exhausted.
        logger.error(
            "[%s] Step '%s' failed after %d retries",
            self.name,
            step.step_name,
            self.max_retries,
        )
        return result

    async def _validate_step(
        self,
        step: PlanStep,
        result: StepResult,
        history: MessageHistory,
    ) -> bool:
        """Ask the LLM (or check programmatically) whether the step passed.

        For structured results with explicit ``passed`` flags, skip LLM.
        Otherwise, ask the LLM to judge against expected_criteria.
        """
        # If the result data has an explicit pass/fail signal, use it.
        if isinstance(result.data, dict):
            if "passed" in result.data:
                return bool(result.data["passed"])
            # Check for error lists.
            errors = result.data.get("errors", [])
            if isinstance(errors, list) and len(errors) == 0:
                return True

        # Check summary for common patterns.
        summary = result.summary
        if summary.get("failed", 0) == 0 and summary.get("total", 0) > 0:
            return True

        # Fallback: ask LLM to judge.
        validation_prompt = (
            f"Step '{step.step_name}' completed with result:\n"
            f"{json.dumps(result.summary, ensure_ascii=False, indent=2)}\n\n"
            f"Expected criteria: {step.expected_criteria}\n\n"
            f"Does the result meet the criteria? "
            f"Answer with a JSON: {{\"passed\": true/false, \"reason\": \"...\"}}"
        )
        history.append(Message(role="system", content=validation_prompt))
        response = await self.llm_client.chat_completions(history.messages)
        history.append(response.message)

        content = response.content or ""
        try:
            # Try to parse structured response.
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            parsed = json.loads(text)
            return bool(parsed.get("passed", False))
        except (json.JSONDecodeError, AttributeError):
            # If parsing fails, check for affirmative keywords.
            lower = content.lower()
            return "true" in lower or "passed" in lower or "yes" in lower
