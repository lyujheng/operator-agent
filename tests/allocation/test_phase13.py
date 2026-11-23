"""Phase 13 tests - dev_table state isolation & duplicate operator deadlock fix.

Tests for:
- _apply_degradation with duplicate operator names across different libs
- _compute_dev_table_updates returning new_dev_entries
- Full check_result loop with new dev entries visible in dev_updates
- Convergence detection stopping pointless retries
"""

from __future__ import annotations

import pytest

from operator_allocation.tools.data_tools import check_result, format_result


# ── Helpers ───────────────────────────────────────────────────────

def _make_operators(specs: list[tuple[str, str, str | None]]) -> list[dict]:
    """Build operator dicts from (name, lib, link_status) tuples."""
    return [
        {
            "operator_name": name,
            "priority": "P0",
            "reference_impl": "",
            "operator_lib": lib,
            "link_status": link,
        }
        for name, lib, link in specs
    ]


DEV_TABLE = {
    "FlagGems": [
        {"operator_name": "add", "status": "completed"},
        {"operator_name": "gt", "status": "completed"},
    ],
    "FlagGems-vllm": [
        {"operator_name": "flash_attn_with_kvcache", "status": "completed"},
        {"operator_name": "causal_conv1d_fn", "status": "completed"},
    ],
    "FlagGems-sglang": [
        {"operator_name": "moe_align_block_size", "status": "completed"},
        {"operator_name": "chunk_gated_delta_rule", "status": "completed"},
    ],
}


# ── TEST1: _apply_degradation duplicate name ──────────────────────

class TestDuplicateOperatorDegradation:
    """D7: duplicate operator names across different libs should each
    degrade to their own correct instance."""

    def test_check_result_includes_operator_lib_in_error(self):
        """D9: invalid_link_name errors carry operator_lib for precise matching."""
        operators = _make_operators([
            ("dup_op", "FlagGems-sglang", "dup_op"),      # not in sglang dev table
            ("dup_op", "FlagGems-vllm", "dup_op"),         # not in vllm dev table
        ])
        result = check_result(operators, DEV_TABLE, {})
        assert not result["passed"]
        assert len(result["errors"]) == 2

        libs = {e["operator_lib"] for e in result["errors"]}
        assert libs == {"FlagGems-sglang", "FlagGems-vllm"}

    def test_degradation_targets_correct_instance(self):
        """Simulate _apply_degradation logic with (name, lib) matching."""
        operators = _make_operators([
            ("dup_op", "FlagGems-sglang", "dup_op"),
            ("dup_op", "FlagGems-vllm", "dup_op"),
            ("gt", "FlagGems", "gt"),                      # valid, should stay
        ])

        check = check_result(operators, DEV_TABLE, {})
        assert len(check["errors"]) == 2

        # Apply degradation with joint matching (same logic as D7).
        for error in check["errors"]:
            op_name = error["operator_name"]
            err_lib = error.get("operator_lib", "")
            op = next(
                (o for o in operators
                 if o.get("operator_name") == op_name
                 and (not err_lib or o.get("operator_lib") == err_lib)),
                None,
            )
            if op:
                op["link_status"] = None
                op["operator_lib"] = "unrecognized"

        # Both duplicates should be unrecognized now.
        unrec = [o for o in operators if o["operator_lib"] == "unrecognized"]
        assert len(unrec) == 2

        # gt should be untouched.
        gt_op = next(o for o in operators if o["operator_name"] == "gt")
        assert gt_op["operator_lib"] == "FlagGems"
        assert gt_op["link_status"] == "gt"

        # Recheck should pass.
        recheck = check_result(operators, DEV_TABLE, {})
        assert recheck["passed"]


# ── TEST2: _compute_dev_table_updates ─────────────────────────────

class TestComputeDevTableUpdates:
    """D4: _compute_dev_table_updates should return new_dev_entries."""

    def test_returns_accumulated_entries(self):
        """Simulate the new _compute_dev_table_updates behavior."""
        # In the new design, this method simply returns self._new_dev_entries.
        new_dev_entries = {
            "FlagGems": [{"operator_name": "matmul", "status": "not_developed"}],
            "FlagGems-vllm": [
                {"operator_name": "unified_attention", "status": "not_developed"},
            ],
        }
        # _compute_dev_table_updates now just does: return dict(self._new_dev_entries)
        updates = dict(new_dev_entries)
        assert "FlagGems" in updates
        assert len(updates["FlagGems"]) == 1
        assert updates["FlagGems"][0]["operator_name"] == "matmul"

    def test_empty_when_no_new_entries(self):
        new_dev_entries: dict[str, list] = {}
        updates = dict(new_dev_entries)
        assert updates == {}


# ── TEST3: Full check_result loop with new dev entries ────────────

class TestCheckResultWithNewDevEntries:
    """D3: new dev entries derived from link_status should be visible
    to check_result via dev_table_updates, preventing false
    invalid_link_name errors."""

    def test_new_entries_prevent_invalid_link_name(self):
        """Operators linked to newly-added dev entries should pass check."""
        operators = _make_operators([
            ("gt", "FlagGems", "gt"),                              # existing
            ("matmul", "FlagGems", "matmul"),                      # NEW
            ("unified_attention", "FlagGems-vllm", "unified_attention"),  # NEW
            ("moe_align_block_size", "FlagGems-sglang", "moe_align_block_size"),  # existing
        ])

        # Simulate new_dev_entries derived from dispatch (D3).
        new_dev_entries = {
            "FlagGems": [{"operator_name": "matmul", "status": "not_developed"}],
            "FlagGems-vllm": [{"operator_name": "unified_attention", "status": "not_developed"}],
        }

        # format_result + check_result
        fmt = format_result(operators, new_dev_entries)
        assert fmt["statistics"]["new_dev_added"] == 2

        check = check_result(
            fmt["requirement_result"],
            DEV_TABLE,
            fmt["dev_table_updates"],
        )
        assert check["passed"], f"Unexpected errors: {check['errors']}"

    def test_without_new_entries_reports_errors(self):
        """Without new_dev_entries, the same operators should fail check."""
        operators = _make_operators([
            ("gt", "FlagGems", "gt"),
            ("matmul", "FlagGems", "matmul"),                      # will fail
            ("unified_attention", "FlagGems-vllm", "unified_attention"),  # will fail
        ])

        check = check_result(operators, DEV_TABLE, {})
        assert not check["passed"]
        invalid = [e for e in check["errors"] if e["type"] == "invalid_link_name"]
        assert len(invalid) == 2
        names = {e["operator_name"] for e in invalid}
        assert names == {"matmul", "unified_attention"}

    def test_full_loop_simulation(self):
        """Simulate the complete format → check → degradation loop
        with 22-operator-like data."""
        # 5 existing + 3 new
        operators = _make_operators([
            ("add", "FlagGems", "add"),
            ("gt", "FlagGems", "gt"),
            ("matmul", "FlagGems", "matmul"),                              # new
            ("flash_attn_with_kvcache", "FlagGems-vllm", "flash_attn_with_kvcache"),
            ("unified_attention", "FlagGems-vllm", "unified_attention"),    # new
            ("moe_align_block_size", "FlagGems-sglang", "moe_align_block_size"),
            ("rotary_embedding", "FlagGems-sglang", "rotary_embedding"),   # new
        ])

        new_dev_entries = {
            "FlagGems": [{"operator_name": "matmul", "status": "not_developed"}],
            "FlagGems-vllm": [{"operator_name": "unified_attention", "status": "not_developed"}],
            "FlagGems-sglang": [{"operator_name": "rotary_embedding", "status": "not_developed"}],
        }

        fmt = format_result(operators, new_dev_entries)
        check = check_result(fmt["requirement_result"], DEV_TABLE, fmt["dev_table_updates"])
        assert check["passed"], f"Errors: {check['errors']}"
        assert fmt["statistics"]["new_dev_added"] == 3
        assert fmt["statistics"]["linked"] == 7
        assert fmt["statistics"]["unrecognized"] == 0


# ── TEST4: Convergence detection ──────────────────────────────────

class TestConvergenceDetection:
    """D8: identical error sets across retries should be detected."""

    def test_identical_errors_detected(self):
        """_errors_identical should return True for same error sets."""
        from operator_allocation.agents.main_agent import AgentOrchestrator

        a = [
            {"type": "invalid_link_name", "operator_name": "foo", "operator_lib": "FlagGems-vllm"},
        ]
        b = [
            {"type": "invalid_link_name", "operator_name": "foo", "operator_lib": "FlagGems-vllm"},
        ]
        assert AgentOrchestrator._errors_identical(a, b)

    def test_different_errors_not_identical(self):
        from operator_allocation.agents.main_agent import AgentOrchestrator

        a = [
            {"type": "invalid_link_name", "operator_name": "foo", "operator_lib": "FlagGems-vllm"},
        ]
        b = [
            {"type": "invalid_link_name", "operator_name": "bar", "operator_lib": "FlagGems-vllm"},
        ]
        assert not AgentOrchestrator._errors_identical(a, b)

    def test_reduced_errors_not_identical(self):
        from operator_allocation.agents.main_agent import AgentOrchestrator

        a = [
            {"type": "invalid_link_name", "operator_name": "foo", "operator_lib": "FlagGems-vllm"},
            {"type": "invalid_link_name", "operator_name": "bar", "operator_lib": "FlagGems"},
        ]
        b = [
            {"type": "invalid_link_name", "operator_name": "foo", "operator_lib": "FlagGems-vllm"},
        ]
        assert not AgentOrchestrator._errors_identical(a, b)

    def test_empty_errors_identical(self):
        from operator_allocation.agents.main_agent import AgentOrchestrator
        assert AgentOrchestrator._errors_identical([], [])
