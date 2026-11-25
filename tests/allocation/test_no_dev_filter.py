"""No-dev filter unit + integration tests (Phase 16 + Phase 17 MCP compat).

Tests for:
  - load_no_dev_known_list: markdown parsing
  - filter_no_dev_operators: Tier 1 exact match, Tier 2 LLM batch classify
  - append_to_pending_list: dedup append
  - NoDevBatchResult schema validation
  - format_result / check_result skip_dev adaptations
  - Orchestrator integration: Step 2 → Step 3 skip_dev filtering
  - Module-level state: set_filter_context / get_filter_context (Phase 17)
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from operator_allocation.models.no_dev import (
    NoDevBatchResult,
    NoDevFilterProfile,
    NoDevJudgment,
)
from operator_allocation.models.operator import RequirementOperator
from operator_allocation.tools.data_tools import check_result, format_result
from operator_allocation.tools.no_dev_filter import (
    _chunk,
    _normalize_category,
    append_to_pending_list,
    filter_no_dev_operators,
    get_filter_context,
    load_no_dev_known_list,
    llm_classify_batch,
    set_filter_context,
)


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture
def sample_known_list_content() -> str:
    return """\
# PyTorch 无需开发算子

共 3 个算子。

## 视图/形状变换（2）

| # | 算子名 |
|---|--------|
| 1 | view |
| 2 | reshape |

## 属性查询（1）

| # | 算子名 |
|---|--------|
| 1 | numel |
"""


@pytest.fixture
def known_list_file(tmp_path: Path, sample_known_list_content: str) -> str:
    path = tmp_path / "known_list.md"
    path.write_text(sample_known_list_content, encoding="utf-8")
    return str(path)


@pytest.fixture
def empty_pending_file(tmp_path: Path) -> str:
    path = tmp_path / "pending_list.md"
    path.write_text(
        "# Pending\n\n<!-- auto -->\n",
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def profile(known_list_file: str, empty_pending_file: str, tmp_path: Path) -> NoDevFilterProfile:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("You are a classifier.", encoding="utf-8")
    return NoDevFilterProfile(
        lib_name="FlagGems",
        known_list_path=known_list_file,
        pending_list_path=empty_pending_file,
        prompt_path=str(prompt_path),
        llm_batch_size=2,
        confidence_threshold="high",
    )


def _make_operators(names: list[str], lib: str = "FlagGems") -> list[dict[str, Any]]:
    """Helper to create operator dicts."""
    return [
        {
            "operator_name": name,
            "priority": "P0",
            "reference_impl": "",
            "operator_lib": lib,
            "link_status": None,
        }
        for name in names
    ]


# ── load_no_dev_known_list ─────────────────────────────────────

class TestLoadNoDevKnownList:
    def test_normal_file(self, known_list_file: str) -> None:
        result = load_no_dev_known_list(known_list_file)
        assert len(result) == 3
        assert result["view"] == "view_shape"
        assert result["reshape"] == "view_shape"
        assert result["numel"] == "attribute_query"

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.md"
        path.write_text("# Empty\n", encoding="utf-8")
        result = load_no_dev_known_list(str(path))
        assert result == {}

    def test_nonexistent_file(self) -> None:
        result = load_no_dev_known_list("/nonexistent/path.md")
        assert result == {}

    def test_format_error_graceful(self, tmp_path: Path) -> None:
        """Malformed markdown should not crash, just parse what it can."""
        path = tmp_path / "bad.md"
        path.write_text("## Bad Category\n\nrandom text\n| 1 | valid_op |\n", encoding="utf-8")
        result = load_no_dev_known_list(str(path))
        assert "valid_op" in result


# ── _normalize_category ────────────────────────────────────────

class TestNormalizeCategory:
    def test_known_categories(self) -> None:
        assert _normalize_category("视图/形状变换") == "view_shape"
        assert _normalize_category("属性查询") == "attribute_query"
        assert _normalize_category("平台专属实现") == "platform_specific"
        assert _normalize_category("Copy 变体") == "copy_variant"
        assert _normalize_category("内存管理") == "memory_management"
        assert _normalize_category("稀疏张量构造") == "sparse_construction"
        assert _normalize_category("Autograd 控制") == "autograd_control"
        assert _normalize_category("符号/断言") == "symbolic_validation"
        assert _normalize_category("测试/内部工具") == "test_internal"

    def test_unknown_category(self) -> None:
        assert _normalize_category("未知类别") == "unknown"


# ── filter_no_dev_operators: Tier 1 ───────────────────────────

class TestFilterTier1:
    def test_exact_match_hit(self, profile: NoDevFilterProfile) -> None:
        """Operators in known list should be marked skip_dev=True."""
        ops = _make_operators(["view", "reshape", "matmul"])
        llm_client = AsyncMock()

        # LLM will be called for matmul only.
        llm_client.get_structured_output = AsyncMock(
            return_value=NoDevBatchResult(judgments=[
                NoDevJudgment(
                    operator_name="matmul",
                    verdict="needs_dev",
                    category=None,
                    confidence="high",
                    reasoning="matrix multiply launches kernel",
                ),
            ])
        )

        stats = asyncio.run(filter_no_dev_operators(ops, profile, llm_client))

        # view, reshape should be exact match.
        assert ops[0]["skip_dev"] is True
        assert ops[0]["skip_reason"].startswith("exact_match:")
        assert ops[1]["skip_dev"] is True

        # matmul should not be skipped.
        assert ops[2]["skip_dev"] is False

        assert stats["tier1_no_dev"] == 2
        assert stats["tier2_no_dev"] == 0
        assert stats["pass_to_dispatch"] == 1

    def test_no_match(self, profile: NoDevFilterProfile) -> None:
        """Operators not in known list go to Tier 2."""
        ops = _make_operators(["relu"])
        llm_client = AsyncMock()
        llm_client.get_structured_output = AsyncMock(
            return_value=NoDevBatchResult(judgments=[
                NoDevJudgment(
                    operator_name="relu",
                    verdict="needs_dev",
                    category=None,
                    confidence="high",
                    reasoning="element-wise op",
                ),
            ])
        )

        stats = asyncio.run(filter_no_dev_operators(ops, profile, llm_client))
        assert stats["tier1_no_dev"] == 0
        assert ops[0]["skip_dev"] is False


# ── filter_no_dev_operators: Tier 2 ───────────────────────────

class TestFilterTier2:
    def test_llm_no_dev_high_confidence(self, profile: NoDevFilterProfile) -> None:
        """LLM verdict=no_dev + confidence=high → skip_dev=True."""
        ops = _make_operators(["_foobar"])
        llm_client = AsyncMock()
        llm_client.get_structured_output = AsyncMock(
            return_value=NoDevBatchResult(judgments=[
                NoDevJudgment(
                    operator_name="_foobar",
                    verdict="no_dev",
                    category="test_internal",
                    confidence="high",
                    reasoning="test stub",
                ),
            ])
        )

        stats = asyncio.run(filter_no_dev_operators(ops, profile, llm_client))
        assert ops[0]["skip_dev"] is True
        assert ops[0]["skip_reason"] == "llm:test_internal"
        assert stats["tier2_no_dev"] == 1

    def test_llm_needs_dev(self, profile: NoDevFilterProfile) -> None:
        """LLM verdict=needs_dev → skip_dev=False."""
        ops = _make_operators(["softmax"])
        llm_client = AsyncMock()
        llm_client.get_structured_output = AsyncMock(
            return_value=NoDevBatchResult(judgments=[
                NoDevJudgment(
                    operator_name="softmax",
                    verdict="needs_dev",
                    category=None,
                    confidence="high",
                    reasoning="compute kernel",
                ),
            ])
        )

        stats = asyncio.run(filter_no_dev_operators(ops, profile, llm_client))
        assert ops[0]["skip_dev"] is False
        assert stats["tier2_no_dev"] == 0

    def test_llm_low_confidence_fallback(self, profile: NoDevFilterProfile) -> None:
        """LLM no_dev + medium confidence → conservative fallback to needs_dev."""
        ops = _make_operators(["some_op"])
        llm_client = AsyncMock()
        llm_client.get_structured_output = AsyncMock(
            return_value=NoDevBatchResult(judgments=[
                NoDevJudgment(
                    operator_name="some_op",
                    verdict="no_dev",
                    category="memory_management",
                    confidence="medium",
                    reasoning="probably no kernel",
                ),
            ])
        )

        stats = asyncio.run(filter_no_dev_operators(ops, profile, llm_client))
        assert ops[0]["skip_dev"] is False  # conservative: not high confidence

    def test_llm_failure_defaults_to_needs_dev(self, profile: NoDevFilterProfile) -> None:
        """LLM failure → all operators in batch default to needs_dev."""
        ops = _make_operators(["op_a", "op_b"])
        llm_client = AsyncMock()
        llm_client.get_structured_output = AsyncMock(side_effect=Exception("API error"))

        stats = asyncio.run(filter_no_dev_operators(ops, profile, llm_client))
        assert all(op["skip_dev"] is False for op in ops)
        assert stats["tier2_no_dev"] == 0


# ── append_to_pending_list ─────────────────────────────────────

class TestAppendToPendingList:
    def test_first_append(self, tmp_path: Path) -> None:
        path = str(tmp_path / "pending.md")
        Path(path).write_text("# Pending\n", encoding="utf-8")

        entries = [
            {"operator_name": "new_op", "category": "test_internal", "reasoning": "test"},
        ]
        count = append_to_pending_list(path, entries)
        assert count == 1

        content = Path(path).read_text(encoding="utf-8")
        assert "new_op" in content

    def test_dedup(self, known_list_file: str) -> None:
        """Entries already in the file should be skipped."""
        entries = [
            {"operator_name": "view", "category": "view_shape", "reasoning": "dup"},
            {"operator_name": "brand_new", "category": "test_internal", "reasoning": "new"},
        ]
        count = append_to_pending_list(known_list_file, entries)
        assert count == 1  # Only brand_new should be appended

    def test_empty_entries(self, empty_pending_file: str) -> None:
        count = append_to_pending_list(empty_pending_file, [])
        assert count == 0


# ── NoDevBatchResult schema ────────────────────────────────────

class TestNoDevBatchResultSchema:
    def test_valid_schema(self) -> None:
        result = NoDevBatchResult(judgments=[
            NoDevJudgment(
                operator_name="test_op",
                verdict="no_dev",
                category="view_shape",
                confidence="high",
                reasoning="pure view",
            ),
        ])
        assert len(result.judgments) == 1
        assert result.judgments[0].verdict == "no_dev"

    def test_invalid_verdict_raises(self) -> None:
        with pytest.raises(Exception):
            NoDevJudgment(
                operator_name="test",
                verdict="invalid",  # type: ignore[arg-type]
                category=None,
                confidence="high",
                reasoning="test",
            )

    def test_empty_judgments(self) -> None:
        result = NoDevBatchResult(judgments=[])
        assert len(result.judgments) == 0


# ── format_result skip_dev adaptation ──────────────────────────

class TestFormatResultSkipDev:
    def test_skip_dev_in_statistics(self) -> None:
        ops = [
            {"operator_name": "view", "priority": "P0", "reference_impl": "x",
             "operator_lib": "FlagGems", "link_status": None, "skip_dev": True,
             "skip_reason": "exact_match:view_shape"},
            {"operator_name": "relu", "priority": "P0", "reference_impl": "x",
             "operator_lib": "FlagGems", "link_status": "relu_op", "skip_dev": False,
             "skip_reason": None},
            {"operator_name": "unknown", "priority": "P0", "reference_impl": "x",
             "operator_lib": "unrecognized", "link_status": None, "skip_dev": False,
             "skip_reason": None},
        ]
        result = format_result(ops, {})
        stats = result["statistics"]

        assert stats["total"] == 3
        assert stats["identified"] == 2
        assert stats["linked"] == 1  # relu only (skip_dev=True excluded)
        assert stats["unrecognized"] == 1
        assert stats["skip_dev_count"] == 1
        assert stats["skip_dev_detail"]["exact_match"] == 1
        assert stats["skip_dev_detail"]["llm"] == 0

    def test_linked_excludes_skip_dev(self) -> None:
        """skip_dev=True with link_status should not count as linked."""
        ops = [
            {"operator_name": "op1", "priority": "P0", "reference_impl": "x",
             "operator_lib": "FlagGems", "link_status": "some_link", "skip_dev": True,
             "skip_reason": "exact_match:view_shape"},
        ]
        result = format_result(ops, {})
        assert result["statistics"]["linked"] == 0

    def test_missing_skip_dev_defaults_to_false(self) -> None:
        """Missing skip_dev field should be filled with False."""
        ops = [
            {"operator_name": "op1", "priority": "P0", "reference_impl": "x",
             "operator_lib": "FlagGems", "link_status": None},
        ]
        result = format_result(ops, {})
        assert ops[0]["skip_dev"] is False

    def test_consistency_check_with_skip_dev(self) -> None:
        """linked <= identified - skip_dev_count should not trigger warning."""
        ops = [
            {"operator_name": "view", "priority": "P0", "reference_impl": "x",
             "operator_lib": "FlagGems", "link_status": None, "skip_dev": True,
             "skip_reason": "exact_match:view_shape"},
            {"operator_name": "relu", "priority": "P0", "reference_impl": "x",
             "operator_lib": "FlagGems", "link_status": "relu_op", "skip_dev": False,
             "skip_reason": None},
        ]
        result = format_result(ops, {})
        stats = result["statistics"]
        # linked=1, identified=2, skip_dev_count=1, 1 <= 2-1=1 → OK
        mismatch_warnings = [w for w in stats["warnings"] if "mismatch" in w.lower()]
        assert len(mismatch_warnings) == 0


# ── check_result skip_dev adaptation ───────────────────────────

class TestCheckResultSkipDev:
    def test_skip_dev_exempt_from_unlinked(self) -> None:
        """skip_dev=True with link_status=None should NOT be reported as unlinked."""
        req_result = [
            {"operator_name": "view", "operator_lib": "FlagGems",
             "link_status": None, "skip_dev": True},
        ]
        result = check_result(req_result, {}, {})
        assert result["passed"] is True

    def test_non_skip_dev_unlinked_reported(self) -> None:
        """skip_dev=False with link_status=None should be reported as unlinked."""
        req_result = [
            {"operator_name": "relu", "operator_lib": "FlagGems",
             "link_status": None, "skip_dev": False},
        ]
        result = check_result(req_result, {}, {})
        assert result["passed"] is False
        assert any(e["type"] == "unlinked_operator" for e in result["errors"])

    def test_mixed_skip_dev(self) -> None:
        """Only non-skip_dev operators with link_status=None should be reported."""
        req_result = [
            {"operator_name": "view", "operator_lib": "FlagGems",
             "link_status": None, "skip_dev": True},
            {"operator_name": "relu", "operator_lib": "FlagGems",
             "link_status": "relu_impl", "skip_dev": False},
        ]
        dev_table = {"FlagGems": [{"operator_name": "relu_impl"}]}
        result = check_result(req_result, dev_table, {})
        assert result["passed"] is True


# ── RequirementOperator model ─────────────────────────────────

class TestRequirementOperatorSkipDev:
    def test_default_skip_dev_false(self) -> None:
        op = RequirementOperator(operator_name="test")
        assert op.skip_dev is False
        assert op.skip_reason is None

    def test_skip_dev_serialization(self) -> None:
        op = RequirementOperator(
            operator_name="view",
            skip_dev=True,
            skip_reason="exact_match:view_shape",
        )
        d = op.to_dict()
        assert d["skip_dev"] is True
        assert d["skip_reason"] == "exact_match:view_shape"


# ── Utility ────────────────────────────────────────────────────

class TestChunk:
    def test_even_split(self) -> None:
        assert _chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_uneven_split(self) -> None:
        assert _chunk([1, 2, 3], 2) == [[1, 2], [3]]

    def test_empty(self) -> None:
        assert _chunk([], 5) == []

    def test_larger_than_input(self) -> None:
        assert _chunk([1, 2], 10) == [[1, 2]]


# ── Module-Level State (Phase 17 MCP compat) ─────────────────

class TestModuleState:
    """Tests for set_filter_context / get_filter_context module state."""

    def test_set_and_get_context(self, profile: NoDevFilterProfile) -> None:
        ops = _make_operators(["view", "relu"])
        set_filter_context(ops, profile.model_dump(), {"base_url": "http://test"})
        got_ops, got_profile, got_settings = get_filter_context()
        assert len(got_ops) == 2
        assert got_profile is not None
        assert got_profile.lib_name == "FlagGems"
        assert got_settings["base_url"] == "http://test"

    def test_filter_from_module_state_tier1(self, profile: NoDevFilterProfile) -> None:
        """filter_no_dev_operators with no args reads from module state (Tier 1 only)."""
        ops = _make_operators(["view", "reshape", "relu"])
        # Set context with empty llm_settings — Tier 1 only (view, reshape in known list).
        # relu will go to Tier 2 but llm_client=None and no _llm_settings → LLM not created,
        # so we pass a mock llm_client explicitly for the Tier 2 portion.
        llm_client = AsyncMock()
        llm_client.get_structured_output = AsyncMock(
            return_value=NoDevBatchResult(judgments=[
                NoDevJudgment(
                    operator_name="relu",
                    verdict="needs_dev",
                    category=None,
                    confidence="high",
                    reasoning="compute kernel",
                ),
            ])
        )
        set_filter_context(ops, profile.model_dump(), {})

        # Call with explicit llm_client for Tier 2 but operators/profile from state.
        stats = asyncio.run(filter_no_dev_operators(llm_client=llm_client))
        assert stats["tier1_no_dev"] == 2  # view, reshape
        assert stats["pass_to_dispatch"] == 1  # relu
        assert "operators" in stats  # MCP compat: modified operators returned
        assert len(stats["operators"]) == 3

    def test_operators_returned_in_result(self, profile: NoDevFilterProfile) -> None:
        """filter_no_dev_operators result includes operators list for MCP mapping."""
        ops = _make_operators(["view"])
        llm_client = AsyncMock()
        stats = asyncio.run(filter_no_dev_operators(ops, profile, llm_client))
        assert "operators" in stats
        assert len(stats["operators"]) == 1
        assert stats["operators"][0]["operator_name"] == "view"
        assert stats["operators"][0]["skip_dev"] is True
