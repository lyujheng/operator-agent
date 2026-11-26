"""Framework unit tests - PlanSolve agent plan parsing and generation."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_framework.agent.base import MessageHistory
from agent_framework.agent.plan_solve import PlanSolveAgent
from agent_framework.agent.plan_types import ExecutionPlan, PlanStep
from agent_framework.agent.tool_registry import ToolRegistry
from agent_framework.llm.types import LLMResponse, Message, Usage


# ── Fixtures ─────────────────────────────────────────────────────

def _make_agent(**kwargs) -> PlanSolveAgent:
    """Create a PlanSolveAgent with a mock LLM client."""
    defaults = dict(
        name="test_agent",
        llm_client=MagicMock(),
        tools=ToolRegistry(),
        system_prompt="You are a test agent.",
    )
    defaults.update(kwargs)
    return PlanSolveAgent(**defaults)


def _plan_json(steps: list[dict]) -> str:
    return json.dumps({"steps": steps})


VALID_STEP = {
    "step_name": "operator_identify",
    "description": "Identify operators",
    "agent_type": "identify",
    "expected_criteria": "all identified",
}

VALID_STEP_2 = {
    "step_name": "task_dispatch",
    "description": "Dispatch tasks",
    "agent_type": "process",
    "expected_criteria": "all dispatched",
}


# ── _parse_plan tests ────────────────────────────────────────────

class TestParsePlan:
    """Test PlanSolveAgent._parse_plan."""

    def test_raw_json_object(self):
        agent = _make_agent()
        content = _plan_json([VALID_STEP])
        plan = agent._parse_plan(content)
        assert len(plan.steps) == 1
        assert plan.steps[0].step_name == "operator_identify"

    def test_raw_json_array_of_steps(self):
        """When LLM returns a bare array instead of {steps: [...]}."""
        agent = _make_agent()
        content = json.dumps([VALID_STEP, VALID_STEP_2])
        plan = agent._parse_plan(content)
        assert len(plan.steps) == 2
        assert plan.steps[1].step_name == "task_dispatch"

    def test_markdown_fence(self):
        agent = _make_agent()
        content = f"```json\n{_plan_json([VALID_STEP])}\n```"
        plan = agent._parse_plan(content)
        assert len(plan.steps) == 1

    def test_markdown_fence_with_trailing_text(self):
        """The original bug: trailing text after closing fence caused parse failure."""
        agent = _make_agent()
        content = f"```json\n{_plan_json([VALID_STEP])}\n```\n\n以上是执行计划 {{详见文档}}"
        plan = agent._parse_plan(content)
        assert len(plan.steps) == 1
        assert plan.steps[0].step_name == "operator_identify"

    def test_json_in_prose(self):
        agent = _make_agent()
        content = f"Here is the plan:\n{_plan_json([VALID_STEP])}\nPlease review."
        plan = agent._parse_plan(content)
        assert len(plan.steps) == 1

    def test_invalid_json_returns_empty_plan(self):
        agent = _make_agent()
        plan = agent._parse_plan("This is not JSON at all")
        assert len(plan.steps) == 0

    def test_empty_steps_returns_empty_plan(self):
        agent = _make_agent()
        plan = agent._parse_plan('{"steps": []}')
        assert len(plan.steps) == 0

    def test_missing_required_fields_returns_empty_plan(self):
        """PlanStep validation failure should return empty plan, not crash."""
        agent = _make_agent()
        content = json.dumps({"steps": [{"step_name": "op"}]})  # missing required fields
        plan = agent._parse_plan(content)
        assert len(plan.steps) == 0


# ── _generate_plan step_name validation ──────────────────────────

def _mock_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=content),
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        raw_response={},
    )


class TestGeneratePlanValidation:
    """Test _generate_plan step_name validation logic."""

    def test_valid_step_names_accepted(self):
        """LLM plan with all valid step names is accepted."""
        executors = {
            "operator_identify": AsyncMock(),
            "task_dispatch": AsyncMock(),
        }
        agent = _make_agent(step_executors=executors)
        agent.llm_client.chat_completions = AsyncMock(
            return_value=_mock_llm_response(_plan_json([VALID_STEP, VALID_STEP_2]))
        )

        history = MessageHistory("test system prompt")
        history.append(Message(role="user", content="test input"))

        plan = asyncio.run(agent._generate_plan(history))
        assert len(plan.steps) == 2
        assert plan.steps[0].step_name == "operator_identify"

    def test_unknown_step_names_trigger_default_plan(self):
        """LLM plan with unregistered step names falls back to default."""
        executors = {
            "operator_identify": AsyncMock(),
            "task_dispatch": AsyncMock(),
        }
        agent = _make_agent(step_executors=executors)

        bad_step = {
            "step_name": "nonexistent_step",
            "description": "Does not exist",
            "agent_type": "unknown",
            "expected_criteria": "n/a",
        }
        agent.llm_client.chat_completions = AsyncMock(
            return_value=_mock_llm_response(_plan_json([VALID_STEP, bad_step]))
        )

        history = MessageHistory("test system prompt")
        history.append(Message(role="user", content="test input"))

        plan = asyncio.run(agent._generate_plan(history))
        default_plan = PlanSolveAgent._default_plan()
        assert [s.step_name for s in plan.steps] == [s.step_name for s in default_plan.steps]

    def test_empty_llm_response_uses_default_plan(self):
        """LLM returns unparseable text -> fall back to default plan."""
        executors = {"operator_identify": AsyncMock()}
        agent = _make_agent(step_executors=executors)
        agent.llm_client.chat_completions = AsyncMock(
            return_value=_mock_llm_response("I cannot create a plan.")
        )

        history = MessageHistory("test system prompt")
        history.append(Message(role="user", content="test input"))

        plan = asyncio.run(agent._generate_plan(history))
        default_plan = PlanSolveAgent._default_plan()
        assert [s.step_name for s in plan.steps] == [s.step_name for s in default_plan.steps]

    def test_dynamic_step_list_in_prompt(self):
        """Planning prompt should list available steps from step_executors."""
        executors = {
            "step_a": AsyncMock(),
            "step_b": AsyncMock(),
        }
        agent = _make_agent(step_executors=executors)
        agent.llm_client.chat_completions = AsyncMock(
            return_value=_mock_llm_response("no json")
        )

        history = MessageHistory("test system prompt")
        history.append(Message(role="user", content="test input"))

        asyncio.run(agent._generate_plan(history))

        # Check the planning message sent to LLM contains dynamic step names
        call_args = agent.llm_client.chat_completions.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
        planning_msg = [m for m in messages if isinstance(m, Message) and m.role == "system"][-1]
        assert "step_a" in planning_msg.content
        assert "step_b" in planning_msg.content


# ── _default_plan tests ──────────────────────────────────────────

class TestDefaultPlan:
    """Test the fallback default plan."""

    def test_default_plan_has_all_steps(self):
        plan = PlanSolveAgent._default_plan()
        step_names = [s.step_name for s in plan.steps]
        assert step_names == [
            "operator_identify",
            "no_dev_filter",
            "task_dispatch",
            "format_result",
            "check_result",
        ]

    def test_default_plan_dependencies(self):
        plan = PlanSolveAgent._default_plan()
        deps = {s.step_name: s.depends_on for s in plan.steps}
        assert deps["operator_identify"] == []
        assert deps["no_dev_filter"] == ["operator_identify"]
        assert deps["task_dispatch"] == ["no_dev_filter"]
        assert deps["format_result"] == ["task_dispatch"]
        assert deps["check_result"] == ["format_result"]
