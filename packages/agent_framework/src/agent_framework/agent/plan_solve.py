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
from pathlib import Path
from typing import Any, Callable, Awaitable

from agent_framework.agent.base import BaseAgent, MessageHistory
from agent_framework.agent.plan_types import (
    ExecutionPlan,
    PlanStep,
    StepResult,
    StepResultSummarizer,
)
from agent_framework.exceptions import MaxRetriesExceeded
from agent_framework.llm.client import _extract_json_text
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
        plan_persist_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.step_executors = step_executors or {}
        self.summarizers = summarizers or {}
        self.plan_persist_path = plan_persist_path

    async def run(self, input_data: Any) -> dict[str, Any]:
        """Execute the full plan-and-solve loop.

        If ``plan_persist_path`` is set, the plan is saved to disk after
        each step.  On restart, a previously incomplete plan is resumed
        from the last completed step.

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

        # ── Phase 1: Planning (or resume) ────────────────────────
        resumed_plan: ExecutionPlan | None = None
        start_index = 0

        if self.plan_persist_path:
            resumed_plan = ExecutionPlan.load(self.plan_persist_path)

        if resumed_plan and resumed_plan.status == "executing":
            plan = resumed_plan
            start_index = plan.current_step_index
            logger.info(
                "Resuming plan from step %d/%d",
                start_index,
                len(plan.steps),
            )
        else:
            plan = await self._generate_plan(history)

        plan.status = "executing"
        self._persist_plan(plan)

        # ── Phase 2: Step-by-step execution ──────────────────────
        for idx in range(start_index, len(plan.steps)):
            step = plan.steps[idx]
            plan.current_step_index = idx
            step_result = await self._execute_step_with_retries(
                step, history, step_results
            )
            step_results[step.step_name] = step_result
            self._persist_plan(plan)

        plan.status = "completed"
        self._persist_plan(plan)

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

    def _persist_plan(self, plan: ExecutionPlan) -> None:
        """Save plan to disk if persistence is configured."""
        if self.plan_persist_path:
            try:
                plan.save(self.plan_persist_path)
            except Exception:
                logger.warning(
                    "Failed to persist plan to %s",
                    self.plan_persist_path,
                    exc_info=True,
                )

    # ── Phase 1 ──────────────────────────────────────────────────

    async def _generate_plan(self, history: MessageHistory) -> ExecutionPlan:
        """Ask the LLM to produce an ExecutionPlan."""
        with span("plan_generation", span_type="llm_call", agent_name=self.name) as s:
            # Build the available step list from registered executors.
            available_steps = ", ".join(sorted(self.step_executors)) if self.step_executors else "operator_identify, task_dispatch, format_result, check_result"
            planning_msg = Message(
                role="system",
                content=(
                    "Analyze the input and create a step-by-step execution plan. "
                    "Return a JSON object with a 'steps' array. Each step has: "
                    "step_name, description, agent_type, expected_criteria, depends_on (list of step_names). "
                    f"Available step_name values (use ONLY these): {available_steps}."
                ),
            )
            history.append(planning_msg)

            plan, content, usage = await self._request_plan(history)
            plan_source = "llm" if plan.steps else "default"

            # Validate: all step_names must have a registered executor.
            if plan.steps and self.step_executors:
                registered = set(self.step_executors)
                plan_names = {s.step_name for s in plan.steps}
                unknown = plan_names - registered
                if unknown:
                    logger.warning(
                        "LLM plan has unregistered steps %s; using default plan",
                        unknown,
                    )
                    plan = ExecutionPlan()
                    plan_source = "default"

            if not plan.steps:
                plan = self._default_plan()
                plan_source = "default"
                logger.warning("LLM plan was empty; using default plan")

            s.output_data = {
                "content_preview": content[:500],
                "usage": usage,
                "plan_steps": [step.step_name for step in plan.steps],
                "plan_source": plan_source,
            }

        return plan

    async def _request_plan(
        self, history: MessageHistory
    ) -> tuple[ExecutionPlan, str, dict[str, int]]:
        """Request a plan from the LLM and parse the response.

        Returns ``(plan, raw_content, usage_dict)``.  The default
        implementation uses ``chat_completions`` + free-text JSON
        extraction.  Override this method to use
        ``get_structured_output`` when the LLM backend supports it::

            async def _request_plan(self, history):
                data = await self.llm_client.get_structured_output(
                    history.messages, ExecutionPlan,
                )
                if data:
                    return ExecutionPlan(**data), json.dumps(data), {}
                return ExecutionPlan(), "", {}
        """
        response = await self.llm_client.chat_completions(history.messages)
        content = response.content or ""
        history.append(response.message)
        plan = self._parse_plan(content)
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
        return plan, content, usage

    def _parse_plan(self, content: str) -> ExecutionPlan:
        """Extract an ExecutionPlan from LLM text (may contain markdown fences)."""
        text = _extract_json_text(content)

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
                    step_name="no_dev_filter",
                    description="判断已识别算子是否需要开发，标记 skip_dev",
                    agent_type="no_dev_filter",
                    expected_criteria=(
                        "所有已配置算子库的算子必须有 skip_dev 判定（True 或 False）"
                    ),
                    depends_on=["operator_identify"],
                ),
                PlanStep(
                    step_name="task_dispatch",
                    description="将已识别且需要开发的算子分发到对应仓库的子 agent 进行关联",
                    agent_type="process",
                    expected_criteria=(
                        "所有非 unrecognized 且 skip_dev=False 的算子的 "
                        "link_status 为字符串或为空"
                    ),
                    depends_on=["no_dev_filter"],
                ),
                PlanStep(
                    step_name="format_result",
                    description="整理结果数据并生成统计信息",
                    agent_type="result",
                    expected_criteria="statistics 内部一致性通过",
                    depends_on=["task_dispatch"],
                ),
                PlanStep(
                    step_name="check_result",
                    description="校验结果完整性和正确性",
                    agent_type="result",
                    expected_criteria="所有校验通过",
                    depends_on=["format_result"],
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
                            role="user",
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
                        s.output_data = {
                            "passed": False,
                            "summary": result.summary,
                        }
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
        history.append(Message(role="user", content=validation_prompt))

        with span(
            f"validate_{step.step_name}",
            span_type="llm_call",
            agent_name=self.name,
        ) as vs:
            response = await self.llm_client.chat_completions(history.messages)
            history.append(response.message)

            content = response.content or ""
            passed = False
            reason = ""
            try:
                # Try to parse structured response.
                text = _extract_json_text(content)
                parsed = json.loads(text)
                passed = bool(parsed.get("passed", False))
                reason = parsed.get("reason", "")
            except (json.JSONDecodeError, AttributeError):
                # If parsing fails, check for affirmative keywords.
                lower = content.lower()
                passed = "true" in lower or "passed" in lower or "yes" in lower
                reason = content[:200]

            vs.output_data = {
                "step": step.step_name,
                "passed": passed,
                "reason": reason[:300],
                "content_preview": content[:300],
            }
            return passed
