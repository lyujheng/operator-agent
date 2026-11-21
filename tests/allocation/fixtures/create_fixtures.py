"""Generate sample fixture Excel files for tests."""

from pathlib import Path

import openpyxl


def create_sample_requirement_excel(path: str) -> None:
    """Create a minimal requirement Excel with test data."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"

    # Header row.
    ws.append(["operator_name", "priority", "reference_impl"])

    # Sample operators covering all libs + unrecognized.
    ws.append(["add", "P0", "https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/BinaryOps.cpp"])
    ws.append(["mul", "P0", "aten/src/ATen/native/BinaryOps.cpp"])
    ws.append(["flash_attention", "P1", "https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops.py"])
    ws.append(["fused_moe", "P1", "https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe.py"])
    ws.append(["custom_op_xyz", "P2", "internal/ops/custom.py"])

    wb.save(path)


def create_sample_dev_table_excel(path: str) -> None:
    """Create a minimal dev table Excel with per-lib sheets."""
    wb = openpyxl.Workbook()
    # Remove default sheet.
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]

    # FlagGems sheet.
    ws_fg = wb.create_sheet("FlagGems")
    ws_fg.append(["id", "operator_name", "status"])
    ws_fg.append([1, "add", "completed"])
    ws_fg.append([2, "sub", "in_progress"])

    # FlagGems-vllm sheet.
    ws_vllm = wb.create_sheet("FlagGems-vllm")
    ws_vllm.append(["id", "operator_name", "status"])
    ws_vllm.append([1, "flash_attention", "not_developed"])

    # FlagGems-sglang sheet.
    ws_sg = wb.create_sheet("FlagGems-sglang")
    ws_sg.append(["id", "operator_name", "status"])

    wb.save(path)


if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent
    create_sample_requirement_excel(str(fixtures_dir / "sample_requirement.xlsx"))
    create_sample_dev_table_excel(str(fixtures_dir / "sample_dev_table.xlsx"))
    print("Fixture files created.")
