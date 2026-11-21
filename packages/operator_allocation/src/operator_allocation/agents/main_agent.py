"""MainAgent (PlanSolveAgent) and AgentOrchestrator.

This module ties the framework's PlanSolveAgent to the operator allocation
business logic: identify operators, dispatch to per-repo sub-agents,
output results, and validate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
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
from agent_framework.agent.tool_registry import ToolRegistry
from agent_framework.llm.client import LLMClient
from agent_framework.mcp.server import MCPStdioServer
from agent_framework.tracing.context import span

from operator_allocation.config.settings import AppSettings
from operator_allocation.models.operator import ProcessResult
from operator_allocation.tools.data_tools import (
    check_result,
    format_result,
)
from operator_allocation.tools.identify_tools import parse_reference_url
from operator_allocation.tools.link_tools import (
    get_operators,
    set_operators,
    update_operator_field,
)

logger = logging.getLogger(__name__)

# Valid operator_lib values.
VALID_LIBS = {"FlagGems", "FlagGems-vllm", "FlagGems-sglang", "unrecognized"}
REPO_LIBS = {"FlagGems", "FlagGems-vllm", "FlagGems-sglang"}


# ── System Prompts ───────────────────────────────────────────────

MAIN_AGENT_SYSTEM_PROMPT = """\
你是一个需求分配编排器，负责协调多个子 agent 完成算子需求的识别、关联和结果输出。

输入: 需求表(list[dict]) + 开发表(dict[str, list[dict]])
输出格式: 生成结构化 ExecutionPlan JSON，包含 steps 列表

可用步骤类型:
- operator_identify: 算子识别（调用识别 agent 确定每个算子的 operator_lib）
- task_dispatch: 任务分发（按 operator_lib 分发到对应仓库的子 agent 进行关联）
- format_result: 结果整理（内存整理 + statistics 计算）
- check_result: 结果检查（完整性和一致性校验 + 降级处理）

每步检查标准示例:
- operator_identify: "所有算子的 operator_lib 字段必须为 FlagGems / FlagGems-vllm / FlagGems-sglang / unrecognized 之一"
- task_dispatch: "所有非 unrecognized 算子的 link_status 为字符串或为空"
- format_result: "statistics 内部一致性通过"
- check_result: "所有校验通过"

重试规则: 单步最多重试 10 次，超出则记录失败明细并继续下一步。
最终输出: {"requirement_result": [...], "dev_table_updates": {...}, "statistics": {...}}
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
4. 写入 link_status（调用 update_link_status 设置关联的开发算子名称）

仓库搜索策略:
- 先用 search_file_content 搜索算子关键词
- 从返回的多条匹配中判断哪个是定义位置（关注 def/class 关键字、文件路径）
- 再用 read_file 精确读取确认
- 需求算子名可能与仓库中的名称不完全一致

命名规范: operator_name 全部小写，下划线连接(如 mul_tensor)，禁止以下划线开头
去重规则: 新增前必须检查开发表中是否已存在同名算子

可用工具: list_repo_files, search_file_content, read_file, add_dev_operator, update_link_status
输出要求: link_status 为开发表中的算子名称字符串或为空（unrecognized 算子）
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


class FormatResultStepSummarizer(StepResultSummarizer):
    """Summarize Step 3: format_result output."""

    def summarize(self, step: PlanStep, result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            return {"step": step.step_name, "data": str(result)[:500]}
        stats = result.get("statistics", {})
        dev_updates = result.get("dev_table_updates", {})
        return {
            "total": stats.get("total", 0),
            "identified": stats.get("identified", 0),
            "linked": stats.get("linked", 0),
            "new_dev_added": stats.get("new_dev_added", 0),
            "unrecognized": stats.get("unrecognized", 0),
            "dev_table_updates_sheets": [k for k, v in dev_updates.items() if v],
        }


class CheckStepSummarizer(StepResultSummarizer):
    """Summarize Step 4: check_result output."""

    def summarize(self, step: PlanStep, result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            return {"passed": False, "detail": str(result)[:500]}
        errors = result.get("errors", [])
        req_errors = [e for e in errors if e.get("type") in {
            "missing_operator", "lib_mismatch", "invalid_link_name", "unlinked_operator",
        }]
        dev_errors = [e for e in errors if e.get("type") in {
            "orphan_dev_operator", "dev_sheet_mismatch", "dev_naming_violation", "dev_duplicate",
        }]
        return {
            "passed": result.get("passed", False),
            "requirement_errors": req_errors,
            "dev_table_errors": dev_errors,
        }


# Summarizer registry.
SUMMARIZER_REGISTRY: dict[str, StepResultSummarizer] = {
    "operator_identify": IdentifyStepSummarizer(),
    "task_dispatch": DispatchStepSummarizer(),
    "format_result": FormatResultStepSummarizer(),
    "check_result": CheckStepSummarizer(),
}


# ── AgentOrchestrator ────────────────────────────────────────────

class AgentOrchestrator:
    """Top-level orchestrator for the operator allocation workflow.

    Accepts structured data (requirement operators + dev table),
    runs the full plan-and-solve loop, and returns in-memory results.

    Architecture:
      main_agent (PlanSolveAgent, this process)
        ├─ identify tools_agent (ReActAgent + MCP subprocess)
        └─ process tools_agents (ReActAgent + MCP subprocess, one per repo)

    Communication between layers uses MCP stdio transport: each tools_agent
    runs as a child process started via ``operator_allocation.server``,
    and the orchestrator discovers/invokes tools through ``MCPStdioServer``.
    """

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._operators: list[dict[str, Any]] = []
        self._dev_table: dict[str, list[dict[str, Any]]] = {}
        self._original_dev_table: dict[str, list[dict[str, Any]]] = {}
        self._sub_agents: dict[str, MCPStdioServer] = {}
        self._temp_files: list[str] = []
        self._format_result_cache: dict[str, Any] | None = None

    async def run(
        self,
        requirement_operators: list[dict[str, Any]],
        dev_table: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Execute the full allocation workflow.

        Parameters
        ----------
        requirement_operators:
            List of requirement operator dicts, each with at minimum
            ``operator_name``, ``priority``, ``reference_impl``.
        dev_table:
            Dev table data keyed by sheet name, e.g.
            ``{"FlagGems": [{"operator_name": "mul", "status": "completed"}, ...]}``
        """
        self._operators = [
            {
                "operator_name": op.get("operator_name", ""),
                "priority": op.get("priority", ""),
                "reference_impl": op.get("reference_impl", ""),
                "operator_lib": op.get("operator_lib"),
                "link_status": op.get("link_status"),
            }
            for op in requirement_operators
        ]
        self._dev_table = {
            sheet: [dict(row) for row in rows]
            for sheet, rows in dev_table.items()
        }
        # Keep a deep copy of the original dev table for diff later.
        self._original_dev_table = {
            sheet: [dict(row) for row in rows]
            for sheet, rows in dev_table.items()
        }

        logger.info("Starting allocation with %d operators", len(self._operators))
        set_operators(self._operators)

        # ── Create LLM client ────────────────────────────────────
        llm_client = LLMClient(
            base_url=self.settings.llm.base_url,
            api_key=self.settings.llm.api_key,
            model=self.settings.llm.model,
            timeout=self.settings.llm.timeout,
            max_retries=self.settings.llm.max_retries,
            structured_output_mode=self.settings.llm.structured_output_mode,
        )

        # ── Build tool registry (for result tools) ───────────────
        result_tools = ToolRegistry()
        result_tools.register(format_result)
        result_tools.register(check_result)

        # ── Create PlanSolve main agent ──────────────────────────
        step_executors = {
            "operator_identify": self._execute_identify,
            "task_dispatch": self._execute_dispatch,
            "format_result": self._execute_format_result,
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
                f"共 {len(self._operators)} 个需求算子，"
                f"开发表包含 {list(self._dev_table.keys())} 等 sheet"
            )
            return result
        finally:
            await llm_client.close()
            await self._cleanup_sub_agents()
            self._cleanup_temp_files()

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

    async def _execute_format_result(
        self, step: PlanStep, prior_results: dict[str, StepResult]
    ) -> dict[str, Any]:
        """Step 3: In-memory result consolidation with statistics."""
        # Compute dev_table_updates by diffing current vs original dev table.
        dev_table_updates = self._compute_dev_table_updates()

        result = format_result(self._operators, dev_table_updates)
        self._format_result_cache = result
        return result

    async def _execute_check(
        self, step: PlanStep, prior_results: dict[str, StepResult]
    ) -> dict[str, Any]:
        """Step 4: In-memory result validation with degradation strategy."""
        if self._format_result_cache is None:
            # If format_result hasn't been called, compute it first.
            dev_table_updates = self._compute_dev_table_updates()
            self._format_result_cache = format_result(self._operators, dev_table_updates)

        req_result = self._format_result_cache.get("requirement_result", self._operators)
        dev_updates = self._format_result_cache.get("dev_table_updates", {})

        check = check_result(req_result, self._original_dev_table, dev_updates)

        if not check["passed"]:
            # Apply degradation strategies.
            check = self._apply_degradation(check, req_result, dev_updates)

        return check

    # ── Degradation Strategy ────────────────────────────────────

    def _apply_degradation(
        self,
        check: dict[str, Any],
        req_result: list[dict[str, Any]],
        dev_updates: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Apply conservative degradation per error type."""
        warnings: list[str] = []
        stats = self._format_result_cache.get("statistics", {}) if self._format_result_cache else {}

        for error in check.get("errors", []):
            err_type = error.get("type", "")
            op_name = error.get("operator_name", "")

            if err_type == "missing_operator":
                # Add missing operator with unrecognized + None.
                req_result.append({
                    "operator_name": op_name,
                    "priority": "",
                    "reference_impl": "",
                    "operator_lib": "unrecognized",
                    "link_status": None,
                })
                warnings.append(f"missing_operator '{op_name}': added as unrecognized")

            elif err_type == "lib_mismatch":
                # Use update_operator_field to fix grouping.
                op = next((o for o in req_result if o.get("operator_name") == op_name), None)
                if op:
                    op["operator_lib"] = "unrecognized"
                    warnings.append(f"lib_mismatch '{op_name}': moved to unrecognized")

            elif err_type == "invalid_link_name":
                op = next((o for o in req_result if o.get("operator_name") == op_name), None)
                if op:
                    op["link_status"] = None
                    op["operator_lib"] = "unrecognized"
                    warnings.append(f"invalid_link_name '{op_name}': cleared link, moved to unrecognized")

            elif err_type == "unlinked_operator":
                # Step 2 already retried. Just record warning.
                warnings.append(f"unlinked_operator '{op_name}': link_status still None after retries")

            elif err_type == "orphan_dev_operator":
                # Remove orphan from dev_table_updates.
                for sheet_ops in dev_updates.values():
                    sheet_ops[:] = [o for o in sheet_ops if o.get("operator_name") != op_name]
                warnings.append(f"orphan_dev_operator '{op_name}': removed from dev_table_updates")

            elif err_type == "dev_sheet_mismatch":
                # Remove mismatched entry.
                for sheet_ops in dev_updates.values():
                    sheet_ops[:] = [o for o in sheet_ops if o.get("operator_name") != op_name]
                warnings.append(f"dev_sheet_mismatch '{op_name}': removed from dev_table_updates")

            elif err_type == "dev_naming_violation":
                warnings.append(f"dev_naming_violation '{op_name}': naming enforced at add_dev_operator")

            elif err_type == "dev_duplicate":
                # Remove duplicate, point link_status to existing.
                for sheet_ops in dev_updates.values():
                    sheet_ops[:] = [o for o in sheet_ops if o.get("operator_name") != op_name]
                # Find requirement operators linking to this and keep link_status
                # (the existing one in original dev table has the same name).
                warnings.append(f"dev_duplicate '{op_name}': removed from updates, link kept to existing")

        # Record warnings in statistics.
        if self._format_result_cache:
            stat = self._format_result_cache.get("statistics", {})
            stat.setdefault("warnings", []).extend(warnings)

        # Re-run check_result to confirm no residual errors.
        recheck = check_result(req_result, self._original_dev_table, dev_updates)
        if not recheck["passed"]:
            # Record unresolved errors.
            if self._format_result_cache:
                stat = self._format_result_cache.get("statistics", {})
                stat["unresolved_errors"] = recheck["errors"]
            logger.warning(
                "check_result still has %d errors after degradation",
                len(recheck["errors"]),
            )

        return recheck

    # ── Dev Table Updates Diff ──────────────────────────────────

    def _compute_dev_table_updates(self) -> dict[str, list[dict[str, Any]]]:
        """Diff current dev table state vs original to find new operators."""
        updates: dict[str, list[dict[str, Any]]] = {}
        for sheet_name, current_ops in self._dev_table.items():
            original_names = {
                op.get("operator_name", "")
                for op in self._original_dev_table.get(sheet_name, [])
            }
            new_ops = [
                op for op in current_ops
                if op.get("operator_name", "") not in original_names
            ]
            if new_ops:
                updates[sheet_name] = new_ops
        return updates

    # ── MCP Server Management ───────────────────────────────────

    async def _spawn_mcp_server(
        self,
        name: str,
        profile: str,
        *,
        repo_path: str = "",
        sheet_name: str = "",
        dev_data: list[dict[str, Any]] | None = None,
        operators_data: list[dict[str, Any]] | None = None,
    ) -> MCPStdioServer:
        """Spawn an MCP bridge server subprocess and connect.

        Parameters
        ----------
        name:
            Identifier for tracking / cleanup.
        profile:
            Tool profile name (e.g. "identify_server", "process_server").
        repo_path:
            Repository path passed to the server (for repo tools).
        sheet_name:
            Dev table sheet name (for process server).
        dev_data:
            Dev table data for this sheet (injected into subprocess for
            data_tools state init).
        operators_data:
            Requirement operators data for this group (injected into
            subprocess for link_tools state init).
        """
        args = ["-m", "operator_allocation.server", f"--profile={profile}"]

        if repo_path:
            args.append(f"--repo_path={repo_path}")
        if sheet_name:
            args.append(f"--sheet_name={sheet_name}")

        if dev_data is not None:
            dev_path = self._write_temp_json(dev_data)
            args.append(f"--dev_excel_path={dev_path}")

        if operators_data is not None:
            ops_path = self._write_temp_json(operators_data)
            args.append(f"--operators-json={ops_path}")

        server = MCPStdioServer(
            command=sys.executable,
            args=args,
        )
        await server.connect()
        self._sub_agents[name] = server

        logger.info("Spawned MCP server '%s' (profile=%s)", name, profile)
        return server

    def _write_temp_json(self, data: Any) -> str:
        """Write data to a temp JSON file for subprocess state injection."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="operator_data_",
            delete=False,
            encoding="utf-8",
        )
        json.dump(data, tmp, ensure_ascii=False, default=str)
        tmp.close()
        self._temp_files.append(tmp.name)
        return tmp.name

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
                update_operator_field([op], "operator_lib", result["operator_lib"])
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
                structured_output_mode=self.settings.llm.structured_output_mode,
            )

            # Spawn identify MCP server subprocess.
            server = await self._spawn_mcp_server(
                "identify",
                "identify_server",
                operators_data=self._operators,
            )
            identify_tools = await ToolRegistry.from_mcp(server)

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
                await self._cleanup_server("identify")

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
        # If result is already a dict (from structured output).
        if isinstance(text, dict):
            lib = text.get("operator_lib", "")
            if lib in VALID_LIBS:
                return lib

        text = str(text)

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

        # Create sub-agents for all groups.
        await self._create_sub_agents(list(groups.keys()), groups)

        # Process each group with a dedicated MCP subprocess + ReAct agent.
        llm_client = LLMClient(
            base_url=self.settings.llm.base_url,
            api_key=self.settings.llm.api_key,
            model=self.settings.llm.model,
            timeout=self.settings.llm.timeout,
            max_retries=self.settings.llm.max_retries,
            structured_output_mode=self.settings.llm.structured_output_mode,
        )

        try:
            tasks = []
            for lib, ops in groups.items():
                server_name = f"process_{lib}"
                server = self._sub_agents.get(server_name)
                if server is None:
                    logger.warning("No MCP server for lib=%s, skipping", lib)
                    continue
                task = asyncio.create_task(
                    self._process_operators_for_lib(
                        server, lib, ops,
                        self._original_dev_table.get(lib, []),
                        self.settings.repo_lib_mapping.get(lib, ""),
                        llm_client,
                    )
                )
                tasks.append(task)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await llm_client.close()

        return self._operators

    async def _create_sub_agents(
        self,
        operator_libs: list[str],
        operators_by_lib: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Create MCP subprocess servers for each operator_lib group.

        Passes both dev table data and requirement operators to each subprocess.
        """
        for lib in operator_libs:
            server_name = f"process_{lib}"
            if server_name in self._sub_agents:
                continue  # Already running from a previous retry.

            repo_path = self.settings.repo_lib_mapping.get(lib, "")
            sheet_name = lib

            await self._spawn_mcp_server(
                server_name,
                "process_server",
                repo_path=repo_path,
                sheet_name=sheet_name,
                dev_data=self._dev_table.get(lib, []),
                operators_data=operators_by_lib.get(lib, []),
            )

    async def _process_operators_for_lib(
        self,
        mcp_server: MCPStdioServer,
        sheet_name: str,
        operators: list[dict[str, Any]],
        original_dev_ops: list[dict[str, Any]],
        repo_path: str,
        llm_client: LLMClient,
    ) -> list[Any]:
        """Process all operators for one repo (e.g. FlagGems-vllm).

        Uses ToolRegistry.from_mcp() to build tools and injects
        result_schema=ProcessResult for structured output extraction.
        """
        process_tools = await ToolRegistry.from_mcp(mcp_server)
        system_prompt = PROCESS_AGENT_SYSTEM_PROMPT.format(sheet_name=sheet_name)

        # Pre-load dev operators for this sheet via MCP.
        try:
            dev_ops_result = await mcp_server.call_tool(
                "read_dev_operators", {"sheet_name": sheet_name}
            )
            dev_ops_text = "\n".join(
                item.text for item in dev_ops_result.content
                if hasattr(item, "text")
            )
            dev_ops = json.loads(dev_ops_text) if dev_ops_text else []
        except Exception:
            dev_ops = [{"operator_name": op.get("operator_name", ""), "status": op.get("status", "")}
                       for op in original_dev_ops]

        results = []
        for op in operators:
            with span(
                f"process_{op['operator_name']}",
                span_type="agent",
                agent_name=f"{sheet_name}_agent",
            ):
                try:
                    agent = ReActAgent(
                        name=f"{sheet_name}_process_{op['operator_name']}",
                        llm_client=llm_client,
                        tools=process_tools,
                        result_schema=ProcessResult,
                        system_prompt=system_prompt,
                        max_turns=self.settings.agent.process_max_turns,
                        max_tool_result_chars=self.settings.agent.max_tool_result_chars,
                        context_compress_threshold=self.settings.agent.context_compress_threshold,
                    )

                    input_data = {
                        "operator": op,
                        "dev_operators": dev_ops,
                        "repo_path": repo_path,
                        "sheet_name": sheet_name,
                    }

                    result = await agent.run(input_data)

                    # Write back link_status from structured result (O3).
                    if isinstance(result, dict):
                        op["link_status"] = result.get("link_status")
                        # Track newly added dev operators.
                        new_dev = result.get("new_dev_operator")
                        if new_dev:
                            dev_ops.append(new_dev)
                    else:
                        # Fallback: try to extract from text.
                        self._extract_link_from_answer(op, str(result), dev_ops)

                    results.append(result)

                except Exception as exc:
                    logger.warning(
                        "Process failed for '%s' in %s: %s",
                        op["operator_name"],
                        sheet_name,
                        exc,
                    )
                    # O8: Preserve original operator_lib on failure, only clear link_status.
                    op["link_status"] = None

        # NOTE: process servers stay alive until flow ends (requirement:
        # all sub-agents live until cleanup).
        return results

    @staticmethod
    def _extract_link_from_answer(
        op: dict[str, Any],
        text: str,
        dev_ops: list[dict[str, Any]],
    ) -> None:
        """Parse the process agent's final answer to update link_status.

        Falls back to text parsing when structured output is unavailable.
        """
        if op.get("link_status") is not None:
            return

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                link = data.get("link_status")
                if link is not None:
                    op["link_status"] = str(link)
                    return
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # ── Cleanup ──────────────────────────────────────────────────

    async def _cleanup_server(self, name: str) -> None:
        """Shut down a single MCP sub-agent process by name."""
        server = self._sub_agents.pop(name, None)
        if server is not None:
            try:
                await server.cleanup()
                logger.debug("Cleaned up MCP server: %s", name)
            except Exception as exc:
                logger.debug("Cleanup error for %s: %s", name, exc)

    async def _cleanup_sub_agents(self) -> None:
        """Shut down all remaining MCP sub-agent processes."""
        for name in list(self._sub_agents.keys()):
            await self._cleanup_server(name)

    def _cleanup_temp_files(self) -> None:
        """Remove temp files created for subprocess state."""
        import os

        for path in self._temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._temp_files.clear()
