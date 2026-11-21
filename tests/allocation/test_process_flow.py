"""Business integration tests - process flow.

Tests for data models, link tools, data tools, repo tools, and Excel tools.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from operator_allocation.models.operator import (
    DevOperator,
    IdentifyResult,
    ProcessResult,
    RequirementOperator,
    ResultOperator,
)
from operator_allocation.tools.data_tools import (
    add_dev_operator as data_add_dev_operator,
    check_result,
    format_result,
    get_dev_table,
    read_dev_operators as data_read_dev_operators,
    set_dev_table,
)
from operator_allocation.tools.link_tools import (
    get_operators,
    set_operators,
    update_link_status,
    update_operator_field,
)
from operator_allocation.tools.repo_tools import list_repo_files, read_file


# ── Model tests ──────────────────────────────────────────────────

class TestOperatorModels:
    def test_requirement_operator(self):
        op = RequirementOperator(
            operator_name="flash_attention",
            priority="P0",
            reference_impl="vllm/vllm/attention.py",
        )
        assert op.operator_lib is None
        assert op.link_status is None  # str | None
        d = op.to_dict()
        assert d["operator_name"] == "flash_attention"

    def test_requirement_operator_link_status_is_str(self):
        op = RequirementOperator(
            operator_name="relu",
            link_status="softmax_kernel",
        )
        assert op.link_status == "softmax_kernel"
        assert isinstance(op.link_status, str)

    def test_dev_operator_no_id(self):
        """DevOperator no longer has an id field."""
        op = DevOperator(operator_name="softmax", status="completed")
        assert op.operator_name == "softmax"
        assert not hasattr(op, "id") or "id" not in op.model_fields

    def test_result_from_requirement(self):
        req = RequirementOperator(
            operator_name="relu",
            priority="P1",
            reference_impl="pytorch/aten/src/ATen/native/Activation.cpp",
            operator_lib="FlagGems",
        )
        result = ResultOperator.from_requirement(req, link_status="relu_kernel")
        assert result.operator_lib == "FlagGems"
        assert result.link_status == "relu_kernel"

    def test_identify_result_schema(self):
        r = IdentifyResult(
            operator_name="relu",
            operator_lib="FlagGems",
            summary="Identified as FlagGems based on pytorch aten source",
        )
        assert r.operator_lib == "FlagGems"

    def test_process_result_schema(self):
        r = ProcessResult(
            operator_name="relu",
            link_status="relu_forward",
            new_dev_operator={"operator_name": "relu_forward", "sheet_name": "FlagGems"},
            summary="Linked to existing dev operator",
        )
        assert r.link_status == "relu_forward"
        assert r.new_dev_operator["sheet_name"] == "FlagGems"

    def test_process_result_no_link(self):
        r = ProcessResult(
            operator_name="unknown_op",
            link_status=None,
            summary="Could not find matching dev operator",
        )
        assert r.link_status is None
        assert r.new_dev_operator is None


# ── Link tools tests ────────────────────────────────────────────

class TestLinkTools:
    def setup_method(self):
        set_operators([
            {"operator_name": "relu", "operator_lib": None, "link_status": None},
            {"operator_name": "softmax", "operator_lib": None, "link_status": None},
            {"operator_name": "gelu", "operator_lib": None, "link_status": None},
        ])

    def test_update_operator_field(self):
        result = update_operator_field(
            [{"operator_name": "relu"}, {"operator_name": "softmax"}],
            "operator_lib",
            "FlagGems",
        )
        assert len(result) == 2
        ops = get_operators()
        assert ops[0]["operator_lib"] == "FlagGems"
        assert ops[1]["operator_lib"] == "FlagGems"
        assert ops[2]["operator_lib"] is None

    def test_update_link_status_str(self):
        """link_status is now str (dev operator name), not int."""
        result = update_link_status("relu", "relu_forward")
        assert result["updated"] is True
        assert result["link_status"] == "relu_forward"

    def test_update_link_status_none(self):
        result = update_link_status("relu", None)
        assert result["updated"] is True
        assert result["link_status"] is None

    def test_update_link_status_nonexistent(self):
        result = update_link_status("nonexistent_op", "some_name")
        assert result["updated"] is False


# ── Data tools tests ────────────────────────────────────────────

class TestDataTools:
    def setup_method(self):
        set_dev_table({
            "FlagGems": [
                {"operator_name": "relu", "status": "completed"},
                {"operator_name": "softmax", "status": "in_progress"},
            ],
            "FlagGems-vllm": [
                {"operator_name": "flash_attention", "status": "not_developed"},
            ],
        })

    def test_read_dev_operators(self):
        ops = data_read_dev_operators("FlagGems")
        assert len(ops) == 2
        assert ops[0]["operator_name"] == "relu"
        assert ops[0]["status"] == "completed"

    def test_read_dev_operators_missing_sheet(self):
        ops = data_read_dev_operators("NonExistent")
        assert ops == []

    def test_add_dev_operator_new(self):
        result = data_add_dev_operator("FlagGems", "gelu")
        assert result["created"] is True
        assert result["operator_name"] == "gelu"
        assert result["sheet_name"] == "FlagGems"
        assert result["error"] is None
        # Verify it's in the dev table now.
        ops = data_read_dev_operators("FlagGems")
        assert len(ops) == 3

    def test_add_dev_operator_idempotent(self):
        r1 = data_add_dev_operator("FlagGems", "relu")
        assert r1["created"] is False
        assert r1["error"] is None

    def test_add_dev_operator_naming_violation(self):
        result = data_add_dev_operator("FlagGems", "BadName")
        assert result["created"] is False
        assert result["error"] is not None
        assert "Naming violation" in result["error"]

    def test_add_dev_operator_leading_underscore(self):
        result = data_add_dev_operator("FlagGems", "_bad_name")
        assert result["created"] is False
        assert result["error"] is not None

    def test_add_dev_operator_new_sheet(self):
        result = data_add_dev_operator("FlagGems-sglang", "new_op")
        assert result["created"] is True
        assert result["sheet_name"] == "FlagGems-sglang"
        ops = data_read_dev_operators("FlagGems-sglang")
        assert len(ops) == 1

    def test_format_result_basic(self):
        operators = [
            {"operator_name": "relu", "priority": "P0", "reference_impl": "pytorch/...",
             "operator_lib": "FlagGems", "link_status": "relu"},
            {"operator_name": "silu", "priority": "P1", "reference_impl": "vllm/...",
             "operator_lib": "FlagGems-vllm", "link_status": "silu_kernel"},
            {"operator_name": "mystery", "priority": "P2", "reference_impl": "",
             "operator_lib": "unrecognized", "link_status": None},
        ]
        dev_updates = {"FlagGems-vllm": [{"operator_name": "silu_kernel", "status": "not_developed"}]}

        result = format_result(operators, dev_updates)
        stats = result["statistics"]
        assert stats["total"] == 3
        assert stats["identified"] == 2
        assert stats["linked"] == 2
        assert stats["new_dev_added"] == 1
        assert stats["unrecognized"] == 1

    def test_format_result_missing_fields(self):
        operators = [{"operator_name": "relu"}]  # missing fields
        result = format_result(operators, {})
        stats = result["statistics"]
        assert stats["total"] == 1
        assert len(stats["warnings"]) > 0  # should warn about missing fields

    def test_check_result_all_pass(self):
        req_result = [
            {"operator_name": "relu", "operator_lib": "FlagGems", "link_status": "relu"},
        ]
        dev_table = {"FlagGems": [{"operator_name": "relu", "status": "completed"}]}
        dev_updates = {}

        result = check_result(req_result, dev_table, dev_updates)
        assert result["passed"] is True

    def test_check_result_invalid_link_name(self):
        req_result = [
            {"operator_name": "relu", "operator_lib": "FlagGems", "link_status": "nonexistent"},
        ]
        dev_table = {"FlagGems": [{"operator_name": "softmax"}]}
        dev_updates = {}

        result = check_result(req_result, dev_table, dev_updates)
        assert result["passed"] is False
        assert any(e["type"] == "invalid_link_name" for e in result["errors"])

    def test_check_result_unlinked_operator(self):
        req_result = [
            {"operator_name": "relu", "operator_lib": "FlagGems", "link_status": None},
        ]
        dev_table = {"FlagGems": []}
        dev_updates = {}

        result = check_result(req_result, dev_table, dev_updates)
        assert result["passed"] is False
        assert any(e["type"] == "unlinked_operator" for e in result["errors"])

    def test_check_result_unrecognized_no_unlinked(self):
        """unrecognized operators with link_status=None are OK."""
        req_result = [
            {"operator_name": "mystery", "operator_lib": "unrecognized", "link_status": None},
        ]
        result = check_result(req_result, {}, {})
        assert result["passed"] is True

    def test_check_result_orphan_dev_operator(self):
        req_result = [
            {"operator_name": "relu", "operator_lib": "FlagGems", "link_status": "relu"},
        ]
        dev_table = {"FlagGems": [{"operator_name": "relu"}]}
        dev_updates = {
            "FlagGems": [{"operator_name": "orphan_op"}],  # no req links to this
        }

        result = check_result(req_result, dev_table, dev_updates)
        assert result["passed"] is False
        assert any(e["type"] == "orphan_dev_operator" for e in result["errors"])

    def test_check_result_dev_duplicate(self):
        req_result = [
            {"operator_name": "relu", "operator_lib": "FlagGems", "link_status": "softmax"},
        ]
        dev_table = {"FlagGems": [{"operator_name": "softmax"}]}
        dev_updates = {
            "FlagGems": [{"operator_name": "softmax"}],  # already exists in original
        }

        result = check_result(req_result, dev_table, dev_updates)
        assert result["passed"] is False
        assert any(e["type"] == "dev_duplicate" for e in result["errors"])

    def test_check_result_dev_naming_violation(self):
        req_result = [
            {"operator_name": "relu", "operator_lib": "FlagGems", "link_status": "BadName"},
        ]
        dev_table = {"FlagGems": []}
        dev_updates = {
            "FlagGems": [{"operator_name": "BadName"}],
        }

        result = check_result(req_result, dev_table, dev_updates)
        assert result["passed"] is False
        assert any(e["type"] == "dev_naming_violation" for e in result["errors"])


# ── Repo tools tests ────────────────────────────────────────────

class TestRepoTools:
    def test_list_repo_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "src").mkdir()
            (Path(tmpdir) / "src" / "model.py").write_text("# model")
            (Path(tmpdir) / "setup.py").write_text("# setup")

            files = list_repo_files(tmpdir)
            assert len(files) == 2
            assert any("model.py" in f for f in files)

    def test_read_file_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = Path(tmpdir) / "test.py"
            fp.write_text("\n".join(f"line {i}" for i in range(1, 21)))

            result = read_file(str(fp), start_line=5, max_lines=3)
            assert result["total_lines"] == 20
            assert result["range"] == [5, 7]
            assert "line 5" in result["content"]
            assert "line 7" in result["content"]

    def test_read_file_not_found(self):
        result = read_file("/nonexistent/path.py")
        assert "not found" in result["content"].lower()
