"""MainAgent (PlanSolveAgent) and AgentOrchestrator.

This module ties the framework's PlanSolveAgent to the operator allocation
business logic: identify operators, dispatch to per-repo sub-agents,
output results, and validate.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from agent_framework.agent.plan_solve import PlanSolveAgent
from agent_framework.agent.plan_types import (
    PlanStep,
    StepResult,
    StepResultSummarizer,
)
from agent_framework.agent.react import ReActAgent
from agent_framework.agent.tool_registry import ToolRegistry, func_to_schema
from agent_framework.llm.client import LLMClient
from agent_framework.mcp.server import MCPStdioServer
from agent_framework.tracing.context import span

from operator_allocation.config.settings import AppSettings
from operator_allocation.tools.excel_tools import (
    add_dev_operator,
    check_result_excel,
    output_result_excel,
    read_dev_operators,
)
from operator_allocation.tools.identify_tools import parse_reference_url
from operator_allocation.tools.link_tools import (
    get_operators,
    set_operators,
    update_link_status,
    update_operator_field,
)
from operator_allocation.tools.repo_tools import (
    list_repo_files,
    read_file,
    search_file_content,
)

logger = logging.getLogger(__name__)

# Valid operator_lib values.
VALID_LIBS = {"FlagGems", "FlagGems-vllm", "FlagGems-sglang", "unrecognized"}
REPO_LIBS = {"FlagGems", "FlagGems-vllm", "FlagGems-sglang"}


# ── System Prompts ───────────────────────────────────────────────

MAIN_AGENT_SYSTEM_PROMPT = """\
你是一个需求分配编排器，负责协调多个子 agent 完成算子需求的识别、关联和结果输出。

输入: 需求表 Excel 文件路径
输出格式: 生成结构化 ExecutionPlan JSON，包含 steps 列表

可用步骤类型:
- operator_identify: 算子识别（调用识别 agent 确定每个算子的 operator_lib）
- task_dispatch: 任务分发（按 operator_lib 分发到对应仓库的子 agent 进行关联）
- output_result: 结果输出（生成按 operator_lib 分组的 result_excel）
- check_result: 结果检查（校验完整性和正确性）

每步检查标准示例:
- operator_identify: "所有算子的 operator_lib 字段必须为 FlagGems / FlagGems-vllm / FlagGems-sglang / unrecognized 之一"
- task_dispatch: "所有算子的 link_status 为数字 id 或为空"
- output_result: "result_excel 文件已生成"
- check_result: "所有校验通过"

重试规则: 单步最多重试 10 次，超出则记录失败明细并继续下一步。
最终输出: result_excel 文件路径 + 处理统计（总数/成功/失败）
"""

IDENTIFY_AGENT_SYSTEM_PROMPT = """\
你是一个算子识别 agent，负责判断单个算子属于哪个源仓库并映射到开发库。

输入: 单个算子（operator_name + reference_impl）、所有源仓库的本地路径

识别策略:
1. 分析 reference_impl 文本，推测源仓库（关键词: torch、aten、vllm、sglang、路径片段等）
2. 有线索时调用 search_file_content 在推测的源仓库中搜索 operator_name 验证
3. 必要时调用 read_file 精确读取确认
4. 验证通过后按映射规则确定 operator_lib

映射规则:
- pytorch (aten 算子), torch_npu → FlagGems
- vllm → FlagGems-vllm
- sglang → FlagGems-sglang

兜底规则: 所有仓库都未找到时，你判断是 aten 算子则标记 FlagGems，否则标记 unrecognized
优先级: reference_impl 指向的仓库优先；未明确时按 aten > vllm > sglang 顺序匹配

可用工具: search_file_content, read_file, list_repo_files, update_operator_field
输出要求: operator_lib 必须为 FlagGems / FlagGems-vllm / FlagGems-sglang / unrecognized 之一，不允许为空
"""

PROCESS_AGENT_SYSTEM_PROMPT = """\
你是 {sheet_name} 仓库的算子处理 agent，负责将单个需求算子与开发表中算子进行关联。

输入: 单个需求算子（已识别 operator_lib）+ 已有开发表算子列表 + 仓库路径

关联逻辑:
1. 先清空当前算子的 link_status（调用 update_link_status 置为 None）
2. 在仓库中确认算子存在（search_file_content 搜索，read_file 确认）
3. 与已有开发表算子匹配（名称匹配），或新增（add_dev_operator）
4. 写入 link_status（调用 update_link_status 设置 id）

仓库搜索策略:
- 先用 search_file_content 搜索算子关键词
- 从返回的多条匹配中判断哪个是定义位置（关注 def/class 关键字、文件路径）
- 再用 read_file 精确读取确认
- 需求算子名可能与仓库中的名称不完全一致

命名规范: operator_name 全部小写，禁止以下划线开头，尽可能与官方仓库算子名一致
去重规则: 新增前必须检查开发表中是否已存在同名算子

可用工具: list_repo_files, search_file_content, read_file, add_dev_operator, update_link_status
输出要求: link_status 为开发表中的数字 id 或为空（unrecognized 算子）
"""


# ── Summarizers ──────────────────────────────────────────────────

class IdentifyStepSummarizer(StepResultSummarizer):
    def summarize(self, step: PlanStep, result: Any) -> dict[str, Any]:
        if not isinstance(result, list):
            return {"step": step.step_name, "data": str(result)[:500]}
        total = len(result)
        failed = [op for op in result if not op.get("operator_lib")]
        libs = Counter(op["operator_lib"] for op in result if op.get("operator_lib"))
        return {
            "total": total,
            "success": total - len(failed),
            "failed": len(failed),
            "summary": dict(libs),
            "failed_operator_names": [op["operator_name"] for op in failed],
        }


class DispatchStepSummarizer(StepResultSummarizer):
    def summarize(self, step: PlanStep, result: Any) -> dict[str, Any]:
        if not isinstance(result, list):
            return {"step": step.step_name, "data": str(result)[:500]}
        total = len(result)
        linked = sum(1 for op in result if op.get("link_status") is not None)
        new_added = sum(1 for op in result if op.get("_newly_added", False))
        failed = total - linked
        return {
            "total": total,
            "linked": linked,
            "new_added": new_added,
            "failed": failed,
            "failed_operator_names": [
                op["operator_name"] for op in result if op.get("link_status") is None
            ],
        }


class OutputStepSummarizer(StepResultSummarizer):
    def summarize(self, step: PlanStep, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return result
        return {"result_path": str(result)}


class CheckStepSummarizer(StepResultSummarizer):
    def summarize(self, step: PlanStep, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return result
        return {"passed": False, "detail": str(result)[:500]}


# Summarizer registry.
SUMMARIZER_REGISTRY: dict[str, StepResultSummarizer] = {
    "operator_identify": IdentifyStepSummarizer(),
    "task_dispatch": DispatchStepSummarizer(),
    "output_result": OutputStepSummarizer(),
    "check_result": CheckStepSummarizer(),
}


# ── AgentOrchestrator ────────────────────────────────────────────

class AgentOrchestrator:
    """Top-level orchestrator for the operator allocation workflow.

    Accepts a requirement Excel, runs the full plan-and-solve loop,
    and outputs result_excel.
    """

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._operators: list[dict[str, Any]] = []
        self._sub_agents: dict[str, MCPStdioServer] = {}

    async def run(self, input_excel_path: str) -> dict[str, Any]:
        """Execute the full allocation workflow."""
        from openpyxl import load_workbook

        # ── Load requirement Excel ───────────────────────────────
        logger.info("Loading requirement Excel: %s", input_excel_path)
        wb = load_workbook(input_excel_path, read_only=True, data_only=True)
        ws = wb.active
        if ws is None:
            raise ValueError(f"No active sheet in {input_excel_path}")

        rows = list(ws.iter_rows(min_row=2, values_only=True))
        wb.close()

        self._operators = []
        for row in rows:
            if not row or not row[0]:
                continue
            self._operators.append({
                "operator_name": str(row[0]).strip(),
                "priority": str(row[1] or "").strip(),
                "reference_impl": str(row[2] or "").strip(),
                "operator_lib": None,
                "link_status": None,
            })

        logger.info("Loaded %d operators from requirement table", len(self._operators))
        set_operators(self._operators)

        # ── Create LLM client ────────────────────────────────────
        llm_client = LLMClient(
            base_url=self.settings.llm.base_url,
            api_key=self.settings.llm.api_key,
            model=self.settings.llm.model,
            timeout=self.settings.llm.timeout,
            max_retries=self.settings.llm.max_retries,
        )

        # ── Build tool registry (for result tools) ───────────────
        result_tools = ToolRegistry()
        result_tools.register(output_result_excel)
        result_tools.register(check_result_excel)

        # ── Create PlanSolve main agent ──────────────────────────
        step_executors = {
            "operator_identify": self._execute_identify,
            "task_dispatch": self._execute_dispatch,
            "output_result": self._execute_output,
            "check_result": self._execute_check,
        }

        main_agent = PlanSolveAgent(
            name="main_agent",
            llm_client=llm_client,
            tools=result_tools,
            system_prompt=MAIN_AGENT_SYSTEM_PROMPT,
            max_retries=self.settings.agent.max_retries,
            step_executors=step_executors,
            summarizers=SUMMARIZER_REGISTRY,
            max_turns=self.settings.agent.max_retries,
        )

        try:
            result = await main_agent.run(
                f"需求表路径: {input_excel_path}, 共 {len(self._operators)} 个算子"
            )
            return result
        finally:
            await llm_client.close()
            await self._cleanup_sub_agents()

    # ── Step Executors ───────────────────────────────────────────

    async def _execute_identify(
        self, step: PlanStep, prior_results: dict[str, StepResult]
    ) -> list[dict[str, Any]]:
        """Step 1: Operator identification (two-phase)."""
        with span("operator_identify", span_type="step", agent_name="main_agent"):
            return await self._identify_operators()

    async def _execute_dispatch(
        self, step: PlanStep, prior_results: dict[str, StepResult]
    ) -> list[dict[str, Any]]:
        """Step 2: Task dispatch to per-repo sub-agents."""
        with span("task_dispatch", span_type="step", agent_name="main_agent"):
            return await self._dispatch_operators()

    async def _execute_output(
        self, step: PlanStep, prior_results: dict[str, StepResult]
    ) -> dict[str, Any]:
        """Step 3: Generate result Excel."""
        output_path = str(
            Path(self.settings.excel.output_dir) / "result.xlsx"
        )
        path = output_result_excel(self._operators, output_path)

        # Count per-sheet.
        libs = Counter(op.get("operator_lib", "unrecognized") for op in self._operators)
        return {
            "result_path": path,
            "sheets": dict(libs),
        }

    async def _execute_check(
        self, step: PlanStep, prior_results: dict[str, StepResult]
    ) -> dict[str, Any]:
        """Step 4: Validate result Excel."""
        result_path = str(Path(self.settings.excel.output_dir) / "result.xlsx")
        return check_result_excel(
            result_path,
            self._operators,
            self.settings.excel.dev_table_path,
        )

    # ── Phase 1: Operator Identification ─────────────────────────

    async def _identify_operators(self) -> list[dict[str, Any]]:
        """Two-phase identification: rule-based then LLM-based."""

        parsed: list[dict[str, Any]] = []
        unparsed: list[dict[str, Any]] = []

        # Phase 1: Rule-based matching.
        for op in self._operators:
            op["operator_lib"] = None  # Reset.
            result = parse_reference_url(op.get("reference_impl", ""))
            if result["operator_lib"]:
                op["operator_lib"] = result["operator_lib"]
                parsed.append(op)
            else:
                unparsed.append(op)

        logger.info(
            "Rule-based identification: %d parsed, %d unparsed",
            len(parsed),
            len(unparsed),
        )

        # Phase 2: LLM-based identification for unparsed operators.
        if unparsed:
            llm_client = LLMClient(
                base_url=self.settings.llm.base_url,
                api_key=self.settings.llm.api_key,
                model=self.settings.llm.model,
                timeout=self.settings.llm.timeout,
                max_retries=self.settings.llm.max_retries,
            )

            identify_tools = ToolRegistry()
            identify_tools.register(search_file_content)
            identify_tools.register(read_file)
            identify_tools.register(list_repo_files)

            try:
                for op in unparsed:
                    with span(
                        f"identify_{op['operator_name']}",
                        span_type="agent",
                        agent_name="identify_agent",
                    ):
                        try:
                            result = await self._identify_single_operator(
                                op, llm_client, identify_tools
                            )
                            op["operator_lib"] = result
                        except Exception as exc:
                            logger.warning(
                                "Identify failed for '%s': %s, marking unrecognized",
                                op["operator_name"],
                                exc,
                            )
                            op["operator_lib"] = "unrecognized"
            finally:
                await llm_client.close()

        return self._operators

    async def _identify_single_operator(
        self,
        op: dict[str, Any],
        llm_client: LLMClient,
        tools: ToolRegistry,
    ) -> str:
        """Run a single-operator ReAct session for identification."""
        repos_info = json.dumps(
            {
                "pytorch": self.settings.repos.pytorch_path,
                "torch_npu": self.settings.repos.torch_npu_path,
                "vllm": self.settings.repos.vllm_path,
                "sglang": self.settings.repos.sglang_path,
            },
            ensure_ascii=False,
        )

        agent = ReActAgent(
            name=f"identify_{op['operator_name']}",
            llm_client=llm_client,
            tools=tools,
            system_prompt=IDENTIFY_AGENT_SYSTEM_PROMPT,
            max_turns=self.settings.agent.identify_max_turns,
            max_tool_result_chars=self.settings.agent.max_tool_result_chars,
            context_compress_threshold=self.settings.agent.context_compress_threshold,
        )

        input_data = {
            "operator_name": op["operator_name"],
            "reference_impl": op.get("reference_impl", ""),
            "repos": repos_info,
        }

        result_text = await agent.run(input_data)

        # Extract operator_lib from the agent's final answer.
        return self._extract_lib_from_answer(result_text or "")

    @staticmethod
    def _extract_lib_from_answer(text: str) -> str:
        """Parse the identify agent's final answer to get operator_lib."""
        # Try JSON first.
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "operator_lib" in data:
                lib = data["operator_lib"]
                if lib in VALID_LIBS:
                    return lib
        except (json.JSONDecodeError, TypeError):
            pass

        # Keyword search in text.
        for lib in ["FlagGems-vllm", "FlagGems-sglang", "FlagGems", "unrecognized"]:
            if lib in text:
                return lib

        return "unrecognized"

    # ── Phase 2: Task Dispatch ───────────────────────────────────

    async def _dispatch_operators(self) -> list[dict[str, Any]]:
        """Dispatch operators to per-repo sub-agents for linking."""

        # Group by operator_lib (skip unrecognized).
        groups: dict[str, list[dict[str, Any]]] = {}
        for op in self._operators:
            lib = op.get("operator_lib", "unrecognized")
            if lib in REPO_LIBS:
                groups.setdefault(lib, []).append(op)

        logger.info("Dispatch groups: %s", {k: len(v) for k, v in groups.items()})

        # Process each group with a dedicated ReAct agent.
        llm_client = LLMClient(
            base_url=self.settings.llm.base_url,
            api_key=self.settings.llm.api_key,
            model=self.settings.llm.model,
            timeout=self.settings.llm.timeout,
            max_retries=self.settings.llm.max_retries,
        )

        try:
            tasks = []
            for lib, ops in groups.items():
                task = asyncio.create_task(
                    self._process_repo_group(lib, ops, llm_client)
                )
                tasks.append(task)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await llm_client.close()

        return self._operators

    async def _process_repo_group(
        self,
        lib: str,
        operators: list[dict[str, Any]],
        llm_client: LLMClient,
    ) -> None:
        """Process all operators for one repo (e.g. FlagGems-vllm)."""
        repo_path = self.settings.repo_lib_mapping.get(lib, "")
        sheet_name = lib

        # Pre-load dev operators for this sheet.
        dev_ops = read_dev_operators(self.settings.excel.dev_table_path, sheet_name)

        # Build tool registry.
        process_tools = ToolRegistry()
        process_tools.register(list_repo_files)
        process_tools.register(search_file_content)
        process_tools.register(read_file)
        process_tools.register(add_dev_operator)
        process_tools.register(update_link_status)

        system_prompt = PROCESS_AGENT_SYSTEM_PROMPT.format(sheet_name=sheet_name)

        # Process each operator independently.
        for op in operators:
            with span(
                f"process_{op['operator_name']}",
                span_type="agent",
                agent_name=f"{lib}_agent",
            ):
                try:
                    agent = ReActAgent(
                        name=f"{lib}_process_{op['operator_name']}",
                        llm_client=llm_client,
                        tools=process_tools,
                        system_prompt=system_prompt,
                        max_turns=self.settings.agent.process_max_turns,
                        max_tool_result_chars=self.settings.agent.max_tool_result_chars,
                        context_compress_threshold=self.settings.agent.context_compress_threshold,
                    )

                    input_data = {
                        "operator": op,
                        "dev_operators": dev_ops,
                        "repo_path": repo_path,
                        "dev_excel_path": self.settings.excel.dev_table_path,
                        "sheet_name": sheet_name,
                    }

                    result_text = await agent.run(input_data)

                    # Try to extract link_status from agent answer.
                    self._extract_link_from_answer(op, result_text or "", dev_ops)

                except Exception as exc:
                    logger.warning(
                        "Process failed for '%s' in %s: %s",
                        op["operator_name"],
                        lib,
                        exc,
                    )
                    op["operator_lib"] = "unrecognized"
                    op["link_status"] = None

    @staticmethod
    def _extract_link_from_answer(
        op: dict[str, Any],
        text: str,
        dev_ops: list[dict[str, Any]],
    ) -> None:
        """Parse the process agent's answer to update link_status."""
        # If link_status was already set by the agent via update_link_status tool,
        # it should already be in the op dict. Only parse if still None.
        if op.get("link_status") is not None:
            return

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                link = data.get("link_status") or data.get("id")
                if link is not None:
                    op["link_status"] = int(link)
                    return
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # ── Cleanup ──────────────────────────────────────────────────

    async def _cleanup_sub_agents(self) -> None:
        """Shut down all MCP sub-agent processes."""
        for name, server in self._sub_agents.items():
            try:
                await server.cleanup()
            except Exception as exc:
                logger.debug("Cleanup error for %s: %s", name, exc)
        self._sub_agents.clear()
