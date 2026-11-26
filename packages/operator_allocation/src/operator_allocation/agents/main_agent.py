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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

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

from operator_allocation.config.no_dev_config import NO_DEV_PROFILES
from operator_allocation.config.settings import AppSettings
from operator_allocation.models.operator import ProcessResult
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
你是一个需求分配编排器（纯编排，不执行具体任务），负责通过 MCP 协调多个 work tool 完成算子需求的识别、过滤、关联和结果输出。

输入: 需求表(list[dict]) + 开发表(dict[str, list[dict]])
输出格式: 生成结构化 ExecutionPlan JSON，包含 steps 列表

可用步骤类型（全部通过 MCP 执行）:
- operator_identify: 算子识别（MCP + ReActAgent，多轮推理）
- no_dev_filter: 无需开发过滤（MCP + LLM-powered function，单次操作）
- task_dispatch: 任务分发（MCP + ReActAgent，每个 lib 独立子进程，skip_dev=True 的算子跳过）
- format_result: 结果整理（MCP + 纯计算 function，字段校验 + statistics 计算）
- check_result: 结果检查（MCP + 纯计算 function，完整性和一致性校验 + 降级处理）

每步检查标准示例:
- operator_identify: "所有算子的 operator_lib 字段必须为 FlagGems / FlagGems-vllm / FlagGems-sglang / unrecognized 之一"
- no_dev_filter: "所有已配置算子库的算子必须有 skip_dev 判定（True 或 False）"
- task_dispatch: "所有非 unrecognized 且 skip_dev=False 的算子的 link_status 为字符串或为空"
- format_result: "statistics 内部一致性通过"
- check_result: "所有校验通过"

重试规则: 单步最多重试 10 次，超出则记录失败明细并继续下一步。
最终输出: {"requirement_result": [...], "dev_table_updates": {...}, "statistics": {...}}
"""

IDENTIFY_AGENT_SYSTEM_PROMPT = """\
你是一个算子识别 agent，负责判断单个算子属于哪个源仓库并映射到开发库。

输入: 单个算子（operator_name + reference_impl）、所有源仓库的名称

识别策略:
1. 分析 reference_impl 文本，推测源仓库（关键词: torch、aten、vllm、sglang、路径片段等）
2. 有线索时调用 search_file_content 在推测的源仓库中搜索 operator_name 验证
3. 必要时调用 read_file 精确读取确认
4. 验证通过后按映射规则确定 operator_lib

关键判断 — 调用点 vs 定义点:
reference_impl URL 指向的位置可能是算子的**调用点**而非**定义点**。
vllm/sglang 中大量使用 PyTorch 内置算子（如 masked_fill_, pow, add, mul, softmax 等），
但这些算子的**实现定义**在 PyTorch/ATen 中，vllm/sglang 只是调用方。

判断方法:
- 如果搜索结果全部是 `tensor.op(...)` / `torch.op(...)` 形式的调用，没有 def/class 定义、
  没有 C++/CUDA kernel 实现、没有 op 注册代码 → 这是调用点，算子实际属于 PyTorch
- 如果搜索到了自定义实现（如 class XxxOp、def xxx_kernel、triton.jit 等）→ 这是定义点，
  算子属于该仓库

纠正规则:
当 reference_impl 指向 vllm/sglang，但经搜索确认该仓库中只有调用点没有自定义实现时，
应将 operator_lib 纠正为 FlagGems（属于 PyTorch/ATen 基础算子）。

映射规则:
- pytorch (aten 算子), torch_npu → FlagGems
- vllm（有自定义实现）→ FlagGems-vllm
- sglang（有自定义实现）→ FlagGems-sglang

兜底规则: 所有仓库都未找到时，你判断是 aten 算子则标记 FlagGems，否则标记 unrecognized
优先级: 定义点所在仓库优先（不是 reference_impl URL 指向的仓库优先）

可用工具: search_file_content, read_file, list_repo_files, update_operator_field
工具使用说明: repo_path 参数传仓库名（如 "vllm"、"pytorch"、"sglang"、"torch_npu"），不要传绝对路径
输出要求: operator_lib 必须为 FlagGems / FlagGems-vllm / FlagGems-sglang / unrecognized 之一，不允许为空
"""

PROCESS_AGENT_SYSTEM_PROMPT = """\
你是 {sheet_name} 仓库的算子处理 agent，负责将单个需求算子与开发表中算子进行关联。

输入: 单个需求算子（已识别 operator_lib）+ 已有开发表算子列表 + sheet 名称

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

工具使用说明: repo_path 参数不需要传，系统会自动注入。调用 search_file_content / read_file / list_repo_files 时省略 repo_path 即可。

命名规范: operator_name 全部小写，下划线连接(如 mul_tensor)，禁止以下划线开头，允许尾随下划线(表示 in-place 操作，如 add_、index_put_impl_)
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


class NoDevFilterStepSummarizer(StepResultSummarizer):
    """Summarize Step 2: no_dev filter output."""

    def summarize(self, step: PlanStep, result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            return {"step": step.step_name, "data": str(result)[:500]}
        return {
            "total": result.get("total", 0),
            "tier1_no_dev": result.get("tier1_no_dev", 0),
            "tier2_no_dev": result.get("tier2_no_dev", 0),
            "pass_to_dispatch": result.get("pass_to_dispatch", 0),
        }


class DispatchStepSummarizer(StepResultSummarizer):
    def summarize(self, step: PlanStep, result: Any) -> dict[str, Any]:
        if not isinstance(result, list):
            return {"step": step.step_name, "data": str(result)[:500]}
        # Only count operators that were dispatched (exclude unrecognized and skip_dev).
        dispatched = [
            op for op in result
            if op.get("operator_lib") in REPO_LIBS and not op.get("skip_dev")
        ]
        total = len(dispatched)
        linked = sum(1 for op in dispatched if op.get("link_status") is not None)
        new_added = sum(1 for op in dispatched if op.get("_newly_added", False))
        failed = total - linked
        skipped_unrecognized = sum(1 for op in result if op.get("operator_lib") == "unrecognized")
        skipped_no_dev = sum(1 for op in result if op.get("skip_dev"))
        return {
            "total": total,
            "linked": linked,
            "new_added": new_added,
            "failed": failed,
            "skipped_unrecognized": skipped_unrecognized,
            "skipped_no_dev": skipped_no_dev,
            "failed_operator_names": [
                op["operator_name"] for op in dispatched if op.get("link_status") is None
            ],
        }


class FormatResultStepSummarizer(StepResultSummarizer):
    """Summarize Step 4: format_result output."""

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
            "skip_dev_count": stats.get("skip_dev_count", 0),
            "dev_table_updates_sheets": [k for k, v in dev_updates.items() if v],
        }


class CheckStepSummarizer(StepResultSummarizer):
    """Summarize Step 5: check_result output."""

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
    "no_dev_filter": NoDevFilterStepSummarizer(),
    "task_dispatch": DispatchStepSummarizer(),
    "format_result": FormatResultStepSummarizer(),
    "check_result": CheckStepSummarizer(),
}


# ── AgentOrchestrator ────────────────────────────────────────────

class AgentOrchestrator:
    """Top-level orchestrator for the operator allocation workflow.

    Accepts structured data (requirement operators + dev table),
    runs the full plan-and-solve loop, and returns in-memory results.

    Architecture (MCP unified execution):
      main_agent (PlanSolveAgent, pure orchestrator — no direct task execution)
        ├─ IDENTIFY_SERVER   (Step 1: ReActAgent, multi-turn reasoning)
        ├─ NO_DEV_SERVER     (Step 2: LLM-powered function, single call)
        ├─ PROCESS_SERVER×N  (Step 3: ReActAgent per lib, multi-turn reasoning)
        └─ RESULT_SERVER     (Step 4+5: pure computation functions, shared)

    All communication uses MCP stdio transport: each work tool runs as
    a child process started via ``operator_allocation.server``, and the
    orchestrator invokes tools through ``MCPStdioServer.call_tool()``.
    """

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._operators: list[dict[str, Any]] = []
        self._dev_table: dict[str, list[dict[str, Any]]] = {}  # read-only after init
        self._new_dev_entries: dict[str, list[dict[str, Any]]] = {}  # new memory for additions
        self._temp_files: list[str] = []
        self._format_result_cache: dict[str, Any] | None = None
        # Shared RESULT_SERVER MCP connection for Step 4/5 (lazy-init).
        self._result_server: MCPStdioServer | None = None
        self._result_server_ctx: Any | None = None  # AsyncContextManager

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
                "skip_dev": op.get("skip_dev", False),
                "skip_reason": op.get("skip_reason"),
            }
            for op in requirement_operators
        ]
        self._dev_table = {
            sheet: [dict(row) for row in rows]
            for sheet, rows in dev_table.items()
        }
        self._new_dev_entries = {}  # reset per run
        self._result_server = None
        self._result_server_ctx = None

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

        # ── Create PlanSolve main agent ──────────────────────────
        # All tools execute through MCP — no local ToolRegistry needed.
        step_executors = {
            "operator_identify": self._execute_identify,
            "no_dev_filter": self._execute_no_dev_filter,
            "task_dispatch": self._execute_dispatch,
            "format_result": self._execute_format_result,
            "check_result": self._execute_check,
        }

        main_agent = PlanSolveAgent(
            name="main_agent",
            llm_client=llm_client,
            tools=ToolRegistry(),  # empty — main agent is pure orchestrator
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

            # Merge business-level data into the framework result.
            # PlanSolveAgent returns {plan, results, statistics (step-level)},
            # but callers (write_result_excel, summary print) expect
            # {requirement_result, dev_table_updates, statistics (business)}.
            if self._format_result_cache:
                result["requirement_result"] = self._format_result_cache.get(
                    "requirement_result", self._operators
                )
                result["dev_table_updates"] = self._format_result_cache.get(
                    "dev_table_updates", {}
                )
                result["statistics"] = self._format_result_cache.get(
                    "statistics", {}
                )
            else:
                # format_result step didn't run or failed; return raw operators.
                result["requirement_result"] = self._operators
                result["dev_table_updates"] = self._compute_dev_table_updates()
                result["statistics"] = {
                    "total": len(self._operators),
                    "identified": sum(1 for o in self._operators if o.get("operator_lib") != "unrecognized"),
                    "linked": sum(1 for o in self._operators if o.get("link_status") is not None and not o.get("skip_dev")),
                    "new_dev_added": 0,
                    "unrecognized": sum(1 for o in self._operators if o.get("operator_lib") == "unrecognized"),
                    "skip_dev_count": sum(1 for o in self._operators if o.get("skip_dev")),
                    "skip_dev_detail": {"exact_match": 0, "llm": 0},
                    "warnings": ["format_result step did not complete"],
                    "unresolved_errors": [],
                }

            return result
        finally:
            await self._shutdown_result_server()
            await llm_client.close()
            self._cleanup_temp_files()

    # ── Step Executors ───────────────────────────────────────────

    async def _execute_identify(
        self, step: PlanStep, prior_results: dict[str, StepResult]
    ) -> list[dict[str, Any]]:
        """Step 1: Operator identification (two-phase)."""
        with span("operator_identify", span_type="step", agent_name="main_agent"):
            return await self._identify_operators()

    async def _execute_no_dev_filter(
        self, step: PlanStep, prior_results: dict[str, StepResult]
    ) -> dict[str, Any]:
        """Step 2: No-dev filter (via MCP NO_DEV_SERVER)."""
        with span("no_dev_filter", span_type="step", agent_name="main_agent"):
            return await self._run_no_dev_filter()

    async def _execute_dispatch(
        self, step: PlanStep, prior_results: dict[str, StepResult]
    ) -> list[dict[str, Any]]:
        """Step 3: Task dispatch to per-repo sub-agents."""
        with span("task_dispatch", span_type="step", agent_name="main_agent"):
            return await self._dispatch_operators()

    async def _execute_format_result(
        self, step: PlanStep, prior_results: dict[str, StepResult]
    ) -> dict[str, Any]:
        """Step 4: Result consolidation via MCP RESULT_SERVER.

        Spawns the shared RESULT_SERVER (kept alive for Step 5).
        """
        dev_table_updates = self._compute_dev_table_updates()
        server = await self._ensure_result_server()

        result = await self._call_result_tool(
            server,
            "format_result",
            {"operators": self._operators, "dev_table_updates": dev_table_updates},
        )
        self._format_result_cache = result
        return result

    async def _execute_check(
        self, step: PlanStep, prior_results: dict[str, StepResult]
    ) -> dict[str, Any]:
        """Step 5: Result validation via MCP RESULT_SERVER (shared with Step 4).

        Degradation logic stays in the main agent (flow control).
        """
        server = await self._ensure_result_server()

        if self._format_result_cache is None:
            dev_table_updates = self._compute_dev_table_updates()
            self._format_result_cache = await self._call_result_tool(
                server,
                "format_result",
                {"operators": self._operators, "dev_table_updates": dev_table_updates},
            )

        req_result = self._format_result_cache.get("requirement_result", self._operators)
        dev_updates = self._format_result_cache.get("dev_table_updates", {})

        check = await self._call_result_tool(
            server,
            "check_result",
            {
                "requirement_result": req_result,
                "dev_table": self._dev_table,
                "dev_table_updates": dev_updates,
            },
        )

        if not check["passed"]:
            prev_errors = check.get("errors", [])
            check = await self._apply_degradation(check, req_result, dev_updates, server)

            # Convergence detection (D8): if recheck errors are identical
            # to pre-degradation errors, the strategy cannot fix them.
            post_errors = check.get("errors", [])
            if not check["passed"] and self._errors_identical(prev_errors, post_errors):
                logger.warning(
                    "Degradation did not reduce errors (%d remaining), "
                    "marking as converged to stop retries",
                    len(post_errors),
                )
                if self._format_result_cache:
                    stat = self._format_result_cache.get("statistics", {})
                    stat["unresolved_errors"] = post_errors
                check["passed"] = True  # stop retry loop

        return check

    @staticmethod
    def _errors_identical(a: list[dict], b: list[dict]) -> bool:
        """Check whether two error lists represent the same set of issues."""
        if len(a) != len(b):
            return False
        key = lambda e: (e.get("type", ""), e.get("operator_name", ""), e.get("operator_lib", ""))
        return sorted(a, key=key) == sorted(b, key=key)

    # ── Degradation Strategy ────────────────────────────────────

    async def _apply_degradation(
        self,
        check: dict[str, Any],
        req_result: list[dict[str, Any]],
        dev_updates: dict[str, list[dict[str, Any]]],
        server: MCPStdioServer,
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
                # D7: use (operator_name, operator_lib) for matching to handle
                # duplicate operator names across different libs.
                err_lib = error.get("operator_lib", "")
                op = next(
                    (o for o in req_result
                     if o.get("operator_name") == op_name
                     and (not err_lib or o.get("operator_lib") == err_lib)),
                    None,
                )
                if op:
                    op["link_status"] = None
                    op["operator_lib"] = "unrecognized"
                    warnings.append(f"invalid_link_name '{op_name}': cleared link, moved to unrecognized")

            elif err_type == "unlinked_operator":
                # Don't auto-reassign: unlinked could be a search failure,
                # not necessarily a misidentified ATen primitive.  The
                # identify-stage prompt improvement handles the root cause.
                warnings.append(f"unlinked_operator '{op_name}': link_status still None after retries")

            elif err_type == "orphan_dev_operator":
                # Remove orphan from the specific lib only.
                err_lib = error.get("operator_lib", "")
                if err_lib and err_lib in dev_updates:
                    sheet_ops = dev_updates[err_lib]
                    sheet_ops[:] = [o for o in sheet_ops if o.get("operator_name") != op_name]
                else:
                    for sheet_ops in dev_updates.values():
                        sheet_ops[:] = [o for o in sheet_ops if o.get("operator_name") != op_name]
                warnings.append(f"orphan_dev_operator '{op_name}': removed from dev_table_updates")

            elif err_type == "dev_lib_mismatch":
                # Remove mismatched entry from the specific lib only.
                err_lib = error.get("operator_lib", "")
                if err_lib and err_lib in dev_updates:
                    sheet_ops = dev_updates[err_lib]
                    sheet_ops[:] = [o for o in sheet_ops if o.get("operator_name") != op_name]
                else:
                    for sheet_ops in dev_updates.values():
                        sheet_ops[:] = [o for o in sheet_ops if o.get("operator_name") != op_name]
                warnings.append(f"dev_lib_mismatch '{op_name}': removed from dev_table_updates")

            elif err_type == "dev_naming_violation":
                warnings.append(f"dev_naming_violation '{op_name}': naming enforced at add_dev_operator")

            elif err_type == "dev_duplicate":
                # Remove duplicate from the specific lib only, keep link_status
                # (the existing one in original dev table has the same name).
                err_lib = error.get("operator_lib", "")
                if err_lib and err_lib in dev_updates:
                    sheet_ops = dev_updates[err_lib]
                    sheet_ops[:] = [o for o in sheet_ops if o.get("operator_name") != op_name]
                else:
                    for sheet_ops in dev_updates.values():
                        sheet_ops[:] = [o for o in sheet_ops if o.get("operator_name") != op_name]
                warnings.append(f"dev_duplicate '{op_name}': removed from updates, link kept to existing")

        # Record warnings in statistics.
        if self._format_result_cache:
            stat = self._format_result_cache.get("statistics", {})
            stat.setdefault("warnings", []).extend(warnings)

        # Re-run check_result via MCP to confirm no residual errors.
        recheck = await self._call_result_tool(
            server,
            "check_result",
            {
                "requirement_result": req_result,
                "dev_table": self._dev_table,
                "dev_table_updates": dev_updates,
            },
        )
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
        """Return new dev entries accumulated during dispatch.

        dev_table is read-only; new entries are tracked separately in
        ``self._new_dev_entries`` and returned as-is here.
        """
        return dict(self._new_dev_entries)

    # ── RESULT_SERVER Lifecycle ─────────────────────────────────

    async def _ensure_result_server(self) -> MCPStdioServer:
        """Lazy-start the shared RESULT_SERVER (Step 4/5 share one process).

        The server stays alive until ``_shutdown_result_server`` is called
        in the ``finally`` block of ``run()``.
        """
        if self._result_server is not None:
            return self._result_server

        ctx = self._mcp_server("result_server")
        server = await ctx.__aenter__()
        self._result_server_ctx = ctx
        self._result_server = server
        logger.info("RESULT_SERVER started (shared for Step 4/5)")
        return server

    async def _shutdown_result_server(self) -> None:
        """Shut down the shared RESULT_SERVER if it's running."""
        if self._result_server_ctx is not None:
            try:
                await self._result_server_ctx.__aexit__(None, None, None)
            except Exception:
                logger.debug("RESULT_SERVER shutdown error (non-fatal)", exc_info=True)
            finally:
                self._result_server = None
                self._result_server_ctx = None
                logger.debug("RESULT_SERVER shut down")

    async def _call_result_tool(
        self,
        server: MCPStdioServer,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a tool on RESULT_SERVER and parse the JSON result."""
        mcp_result = await server.call_tool(tool_name, arguments)
        result_text = ""
        for item in mcp_result.content:
            if hasattr(item, "text"):
                result_text += item.text
        return json.loads(result_text)

    # ── MCP Server Management ───────────────────────────────────

    @asynccontextmanager
    async def _mcp_server(
        self,
        profile: str,
        *,
        repo_path: str = "",
        sheet_name: str = "",
        dev_data: list[dict[str, Any]] | None = None,
        operators_data: list[dict[str, Any]] | None = None,
        filter_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[MCPStdioServer]:
        """Spawn an MCP bridge server subprocess as an async context manager.

        The server's lifetime is bound to the ``async with`` block,
        ensuring that anyio cancel scopes are entered and exited in the
        same task in stack order.

        Parameters
        ----------
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
        filter_context:
            No-dev filter context dict ``{operators, profile, llm_settings}``
            (injected into subprocess for no_dev_filter state init).
        """
        args = ["-m", "operator_allocation.server", f"--profile={profile}"]

        if repo_path:
            args.append(f"--repo_path={repo_path}")
        if sheet_name:
            args.append(f"--sheet_name={sheet_name}")
        # Pass base_dir so identify_server can resolve repo names to absolute
        # paths internally, keeping absolute paths out of LLM messages.
        if profile == "identify_server":
            args.append(f"--repo-base-dir={self.settings.repos.base_dir}")

        if dev_data is not None:
            dev_path = self._write_temp_json(dev_data)
            args.append(f"--dev_excel_path={dev_path}")

        if operators_data is not None:
            ops_path = self._write_temp_json(operators_data)
            args.append(f"--operators-json={ops_path}")

        if filter_context is not None:
            ctx_path = self._write_temp_json(filter_context)
            args.append(f"--filter-context-json={ctx_path}")

        server = MCPStdioServer(
            command=sys.executable,
            args=args,
        )

        async with server:
            logger.info("Spawned MCP server (profile=%s)", profile)
            yield server
        logger.debug("MCP server shut down (profile=%s)", profile)

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

    # ── Phase 1.5: No-Dev Filter ──────────────────────────────────

    async def _run_no_dev_filter(self) -> dict[str, Any]:
        """Execute no_dev filtering for all configured libs via MCP.

        For libs with a NoDevFilterProfile, spawns a NO_DEV_SERVER MCP
        subprocess, injects operators + profile + LLM settings, and calls
        ``filter_no_dev_operators`` through the MCP boundary.

        For unconfigured libs, sets skip_dev=False on all operators.
        """
        combined_stats: dict[str, Any] = {
            "total": 0,
            "tier1_no_dev": 0,
            "tier2_no_dev": 0,
            "pass_to_dispatch": 0,
        }

        # Build profile map, merging config overrides with registered profiles.
        profiles = dict(NO_DEV_PROFILES)
        for lib_name, lib_settings in self.settings.no_dev_filter.items():
            if lib_name in profiles:
                # Override profile fields from config.toml.
                profile = profiles[lib_name]
                if lib_settings.known_list_path:
                    profile = profile.model_copy(
                        update={k: v for k, v in lib_settings.model_dump().items() if v}
                    )
                    profiles[lib_name] = profile

        llm_settings = {
            "base_url": self.settings.llm.base_url,
            "api_key": self.settings.llm.api_key,
            "model": self.settings.llm.model,
            "timeout": self.settings.llm.timeout,
            "max_retries": self.settings.llm.max_retries,
            "structured_output_mode": self.settings.llm.structured_output_mode,
        }

        # Run filtering for each configured lib (with up to 3 retries on
        # incomplete skip_dev judgments per design doc line 238).
        max_filter_retries = 3
        for attempt in range(1, max_filter_retries + 1):
            # Reset stats each attempt so retries don't double-count.
            combined_stats = {
                "total": 0,
                "tier1_no_dev": 0,
                "tier2_no_dev": 0,
                "pass_to_dispatch": 0,
            }
            for lib_name, profile in profiles.items():
                # Select operators for this lib.
                lib_ops = [
                    op for op in self._operators
                    if op.get("operator_lib") == lib_name
                ]
                if not lib_ops:
                    continue

                # model_dump(mode="json") converts set→list for
                # JSON-safe serialisation (valid_categories is set[str]).
                filter_ctx = {
                    "operators": lib_ops,
                    "profile": profile.model_dump(mode="json"),
                    "llm_settings": llm_settings,
                }

                with span(
                    f"no_dev_filter_{lib_name}",
                    span_type="filter",
                    agent_name="main_agent",
                ):
                    async with self._mcp_server(
                        "no_dev_filter_server",
                        filter_context=filter_ctx,
                    ) as server:
                        mcp_result = await server.call_tool(
                            "filter_no_dev_operators", {},
                        )

                        # Extract JSON from MCP CallToolResult.
                        result_text = ""
                        for item in mcp_result.content:
                            if hasattr(item, "text"):
                                result_text += item.text
                        stats = json.loads(result_text)

                        # Map modified operators back to self._operators.
                        # Use (name, lib) compound key to avoid cross-lib
                        # contamination when duplicate names exist (Phase 13).
                        if "operators" in stats:
                            result_map = {
                                (op["operator_name"], op.get("operator_lib", "")): op
                                for op in stats["operators"]
                            }
                            for op in self._operators:
                                key = (op["operator_name"], op.get("operator_lib", ""))
                                if key in result_map:
                                    result_op = result_map[key]
                                    op["skip_dev"] = result_op.get("skip_dev", False)
                                    op["skip_reason"] = result_op.get("skip_reason")

                        for key in combined_stats:
                            combined_stats[key] += stats.get(key, 0)

            # For unconfigured libs: ensure all operators have skip_dev set.
            configured_libs = set(profiles.keys())
            for op in self._operators:
                lib = op.get("operator_lib", "unrecognized")
                if lib not in configured_libs and lib != "unrecognized":
                    op.setdefault("skip_dev", False)
                    op.setdefault("skip_reason", None)

            # Validation: all non-unrecognized operators must have skip_dev.
            missing_judgment = [
                op for op in self._operators
                if op.get("operator_lib") != "unrecognized"
                and "skip_dev" not in op
            ]
            if not missing_judgment:
                break  # All operators have judgments.

            if attempt < max_filter_retries:
                logger.warning(
                    "%d operators missing skip_dev judgment (attempt %d/%d), retrying",
                    len(missing_judgment), attempt, max_filter_retries,
                )
            else:
                # Exceeded retries: default to needs_dev.
                logger.warning(
                    "%d operators missing skip_dev judgment after %d retries, defaulting to False",
                    len(missing_judgment), max_filter_retries,
                )
                for op in missing_judgment:
                    op["skip_dev"] = False
                    op["skip_reason"] = None

        logger.info(
            "No-dev filter complete: %d total, %d tier1, %d tier2, %d pass",
            combined_stats["total"],
            combined_stats["tier1_no_dev"],
            combined_stats["tier2_no_dev"],
            combined_stats["pass_to_dispatch"],
        )
        return combined_stats

    # ── Phase 1: Operator Identification ─────────────────────────

    async def _identify_operators(self) -> list[dict[str, Any]]:
        """Two-phase identification: rule-based then LLM-based."""

        parsed: list[dict[str, Any]] = []
        unparsed: list[dict[str, Any]] = []

        # Phase 1: Rule-based matching.
        # Step 1a: Classify all operators via URL patterns.
        flaggems_ops: list[dict[str, Any]] = []
        # Operators pending grep verification, grouped by lib.
        pending_verify: dict[str, list[dict[str, Any]]] = {}
        no_match: list[dict[str, Any]] = []

        for op in self._operators:
            op["operator_lib"] = None  # Reset.
            result = parse_reference_url(op.get("reference_impl", ""))
            if result["operator_lib"]:
                lib = result["operator_lib"]
                if lib == "FlagGems":
                    # pytorch/aten → direct conclusion, no verification needed.
                    op["operator_lib"] = lib
                    flaggems_ops.append(op)
                else:
                    # vllm/sglang → collect for batch verification.
                    pending_verify.setdefault(lib, []).append(op)
            else:
                no_match.append(op)

        parsed.extend(flaggems_ops)

        # Step 1b: Batch-verify vllm/sglang operators per repo.
        for lib, ops in pending_verify.items():
            repo_path = self.settings.repo_lib_mapping.get(lib, "")
            if not repo_path:
                # No repo configured for this lib; fall through to Phase 2.
                unparsed.extend(ops)
                continue

            op_names = [op["operator_name"] for op in ops]
            found = self._batch_check_definitions(op_names, repo_path)

            if found is None:
                # Batch grep failed (timeout / unavailable); all go to Phase 2.
                logger.warning(
                    "Batch grep failed for %s (%d operators); deferring to Phase 2",
                    lib,
                    len(ops),
                )
                unparsed.extend(ops)
                continue

            for op in ops:
                if op["operator_name"] in found:
                    op["operator_lib"] = lib
                    parsed.append(op)
                else:
                    # Definition not found; let Phase 2 decide.
                    unparsed.append(op)

        unparsed.extend(no_match)

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

            try:
                # Spawn identify MCP server — lifetime bound to this block.
                async with self._mcp_server(
                    "identify_server",
                    operators_data=self._operators,
                ) as server:
                    identify_tools = await ToolRegistry.from_mcp(server)

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
                # Server automatically cleaned up here.
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
        # Pass only repo names, not absolute paths — the identify_server
        # subprocess resolves names to absolute paths via set_repo_base_dir,
        # so absolute paths never enter LLM messages (which would trigger WAF).
        repos_info = json.dumps(
            {
                "pytorch": "pytorch",
                "torch_npu": "torch_npu",
                "vllm": "vllm",
                "sglang": "sglang",
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

        # Group by operator_lib (skip unrecognized and skip_dev=True).
        groups: dict[str, list[dict[str, Any]]] = {}
        for op in self._operators:
            lib = op.get("operator_lib", "unrecognized")
            if lib in REPO_LIBS and not op.get("skip_dev"):
                groups.setdefault(lib, []).append(op)

        logger.info("Dispatch groups: %s", {k: len(v) for k, v in groups.items()})

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
            # Each _dispatch_single_lib spawns its own MCP server via
            # async with, so we can run them concurrently.
            tasks = []
            for lib, ops in groups.items():
                task = asyncio.create_task(
                    self._dispatch_single_lib(
                        lib, ops,
                        self._dev_table.get(lib, []),
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

    async def _dispatch_single_lib(
        self,
        lib: str,
        operators: list[dict[str, Any]],
        original_dev_ops: list[dict[str, Any]],
        repo_path: str,
        llm_client: LLMClient,
    ) -> list[Any]:
        """Spawn an MCP server for one lib and process all its operators.

        The server lifetime is bound to the ``async with`` block — it is
        automatically cleaned up when the block exits.
        """
        async with self._mcp_server(
            "process_server",
            repo_path=repo_path,
            sheet_name=lib,
            dev_data=self._dev_table.get(lib, []),
            operators_data=[op for op in self._operators if op.get("operator_lib") == lib],
        ) as server:
            return await self._process_operators_for_lib(
                server, lib, operators,
                original_dev_ops, repo_path, llm_client,
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
                        # repo_path 不传给 LLM（防止绝对路径进入 HTTP body 触发 WAF）。
                        # process_server 通过 extra_context 自动注入 repo_path 到工具调用。
                        "sheet_name": sheet_name,
                    }

                    result = await agent.run(input_data)

                    # Write back link_status from structured result (O3).
                    if isinstance(result, dict):
                        op["link_status"] = result.get("link_status")
                    else:
                        # Fallback: try to extract from text.
                        self._extract_link_from_answer(op, str(result), dev_ops)

                    # Derive new dev entries from link_status (D3).
                    # If link_status points to a name not in original dev_table
                    # or already-accumulated new_dev_entries, record it.
                    link = op.get("link_status")
                    if link and sheet_name in REPO_LIBS:
                        existing = {o["operator_name"] for o in self._dev_table.get(sheet_name, [])}
                        existing |= {o["operator_name"] for o in self._new_dev_entries.get(sheet_name, [])}
                        if link not in existing:
                            new_entry = {"operator_name": link, "status": "not_developed"}
                            self._new_dev_entries.setdefault(sheet_name, []).append(new_entry)
                            dev_ops.append(new_entry)

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

        # Server lifetime is managed by the caller's async with block.
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

    # ── Repo Definition Verification ──────────────────────────────

    @staticmethod
    def _batch_check_definitions(
        operator_names: list[str],
        repo_path: str,
    ) -> set[str] | None:
        """Check which operators have ``def`` definitions in *repo_path*.

        Runs a **single** ``grep -rlF`` pass over the whole repo with all
        operator names as ``-e`` patterns, then a quick per-name check on
        only the matched files.  This is dramatically faster on WSL2 than
        per-operator grep because the expensive filesystem scan happens once.

        Returns:
            A ``set`` of operator names that have definitions, or ``None``
            if grep failed entirely (timeout / unavailable) — the caller
            should fall through to Phase 2 for all operators in that case.
        """
        import subprocess

        if not operator_names:
            return set()

        # Pass 1: single scan to find all files that contain ANY "def <name>(".
        args = ["grep", "-rlF", "--include=*.py"]
        for name in operator_names:
            args.extend(["-e", f"def {name}("])
        args.append(repo_path)

        try:
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=60,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.warning("Batch grep pass-1 failed for %s: %s", repo_path, exc)
            return None

        matched_files = result.stdout.strip()
        if not matched_files:
            return set()

        files = matched_files.split("\n")

        # Pass 2: per-name check on only the matched files (fast — small set).
        found: set[str] = set()
        for name in operator_names:
            try:
                r = subprocess.run(
                    ["grep", "-lF", f"def {name}("] + files,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if r.stdout.strip():
                    found.add(name)
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                # Individual check failed; treat as not found (→ Phase 2).
                pass

        return found

    # ── Cleanup ──────────────────────────────────────────────────

    def _cleanup_temp_files(self) -> None:
        """Remove temp files created for subprocess state."""
        import os

        for path in self._temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._temp_files.clear()
