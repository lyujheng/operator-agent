"""Business integration tests - process flow.

Tests for Excel tools, link tools, repo tools, and data models.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from operator_allocation.models.operator import (
    DevOperator,
    RequirementOperator,
    ResultOperator,
)
from operator_allocation.tools.excel_tools import (
    add_dev_operator,
    check_result_excel,
    output_result_excel,
    read_dev_operators,
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
        d = op.to_dict()
        assert d["operator_name"] == "flash_attention"

    def test_dev_operator(self):
        op = DevOperator(id=1, operator_name="softmax", status="completed")
        assert op.id == 1

    def test_result_from_requirement(self):
        req = RequirementOperator(
            operator_name="relu",
            priority="P1",
            reference_impl="pytorch/aten/src/ATen/native/Activation.cpp",
            operator_lib="FlagGems",
        )
        result = ResultOperator.from_requirement(req, link_status=42)
        assert result.operator_lib == "FlagGems"
        assert result.link_status == 42


# ── Link tools tests ────────────────────────────────────────────

class TestLinkTools:
    def setup_method(self):
        set_operators([
            {"operator_name": "relu", "operator_lib": None, "link_status": None},
            {"operator_name": "softmax", "operator_lib": None, "link_status": None},
            {"operator_name": "gelu", "operator_lib": None, "link_status": None},
        ])

    def test_update_operator_field(self):
        result = update_operator_field(["relu", "softmax"], "operator_lib", "FlagGems")
        assert len(result) == 2
        ops = get_operators()
        assert ops[0]["operator_lib"] == "FlagGems"
        assert ops[1]["operator_lib"] == "FlagGems"
        assert ops[2]["operator_lib"] is None

    def test_update_link_status(self):
        result = update_link_status("relu", 42)
        assert result["updated"] is True
        assert result["link_status"] == 42

    def test_update_link_status_nonexistent(self):
        result = update_link_status("nonexistent_op", 1)
        assert result["updated"] is False


# ── Excel tools tests ────────────────────────────────────────────

class TestExcelTools:
    def test_add_dev_operator_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "dev.xlsx")

            # First add creates.
            r1 = add_dev_operator(path, "FlagGems", "softmax")
            assert r1["created"] is True
            assert r1["id"] == 1

            # Second add is idempotent.
            r2 = add_dev_operator(path, "FlagGems", "softmax")
            assert r2["created"] is False
            assert r2["id"] == 1

            # Different operator gets new id.
            r3 = add_dev_operator(path, "FlagGems", "relu")
            assert r3["created"] is True
            assert r3["id"] == 2

    def test_add_dev_operator_name_normalization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "dev.xlsx")

            r = add_dev_operator(path, "FlagGems", "_Softmax")
            assert r["operator_name"] == "softmax"  # lowercase, no leading underscore

    def test_read_dev_operators(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "dev.xlsx")
            add_dev_operator(path, "FlagGems", "relu")
            add_dev_operator(path, "FlagGems", "softmax")

            ops = read_dev_operators(path, "FlagGems")
            assert len(ops) == 2
            assert ops[0]["operator_name"] == "relu"
            assert ops[1]["operator_name"] == "softmax"

    def test_read_dev_operators_missing_sheet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "dev.xlsx")
            add_dev_operator(path, "FlagGems", "relu")

            ops = read_dev_operators(path, "NonExistent")
            assert ops == []

    def test_output_result_excel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "result.xlsx")
            operators = [
                {"operator_name": "relu", "priority": "P0", "reference_impl": "pytorch/...", "operator_lib": "FlagGems", "link_status": 1},
                {"operator_name": "silu", "priority": "P1", "reference_impl": "vllm/...", "operator_lib": "FlagGems-vllm", "link_status": 2},
                {"operator_name": "mystery", "priority": "P2", "reference_impl": "", "operator_lib": "unrecognized", "link_status": None},
            ]
            path = output_result_excel(operators, out_path)
            assert Path(path).exists()

    def test_check_result_excel_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup dev table.
            dev_path = str(Path(tmpdir) / "dev.xlsx")
            add_dev_operator(dev_path, "FlagGems", "relu")

            # Setup result excel.
            result_path = str(Path(tmpdir) / "result.xlsx")
            operators = [
                {"operator_name": "relu", "priority": "P0", "reference_impl": "pytorch/...", "operator_lib": "FlagGems", "link_status": 1},
            ]
            output_result_excel(operators, result_path)

            # Check.
            result = check_result_excel(result_path, operators, dev_path)
            assert result["passed"] is True
            assert len(result["errors"]) == 0

    def test_check_result_excel_missing_operator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dev_path = str(Path(tmpdir) / "dev.xlsx")
            add_dev_operator(dev_path, "FlagGems", "relu")

            result_path = str(Path(tmpdir) / "result.xlsx")
            output_result_excel(
                [{"operator_name": "relu", "priority": "P0", "reference_impl": "", "operator_lib": "FlagGems", "link_status": 1}],
                result_path,
            )

            # Requirement has an operator not in result.
            req_ops = [
                {"operator_name": "relu"},
                {"operator_name": "softmax"},  # missing from result
            ]
            result = check_result_excel(result_path, req_ops, dev_path)
            assert result["passed"] is False
            missing = [e for e in result["errors"] if e["type"] == "missing_operator"]
            assert len(missing) == 1
            assert missing[0]["operator_name"] == "softmax"


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
