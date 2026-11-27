"""Business integration tests - identify tools."""

from __future__ import annotations

import pytest

from operator_allocation.tools.identify_tools import configure_patterns, parse_reference_url


class TestParseReferenceUrl:
    """Test the rule-based operator_lib identification."""

    # ── URL-based matching ───────────────────────────────────────

    def test_pytorch_github_url(self):
        result = parse_reference_url(
            "https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp"
        )
        assert result["operator_lib"] == "FlagGems"
        assert result["match_type"] == "url"

    def test_vllm_github_url(self):
        result = parse_reference_url(
            "https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py"
        )
        assert result["operator_lib"] == "FlagGems-vllm"
        assert result["match_type"] == "url"

    def test_sglang_github_url(self):
        result = parse_reference_url(
            "https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention.py"
        )
        assert result["operator_lib"] == "FlagGems-sglang"
        assert result["match_type"] == "url"

    def test_torch_npu_url(self):
        result = parse_reference_url(
            "https://github.com/Ascend/torch_npu/blob/master/npu/core/NPUBridge.cpp"
        )
        assert result["operator_lib"] == "FlagGems"
        assert result["match_type"] == "url"

    def test_unknown_github_url(self):
        result = parse_reference_url(
            "https://github.com/someuser/somerepo/blob/main/file.py"
        )
        assert result["operator_lib"] is None
        assert result["repo_url"] is not None

    # ── Path-based matching ──────────────────────────────────────

    def test_vllm_path(self):
        result = parse_reference_url("vllm/vllm/model_executor/layers/activation.py")
        assert result["operator_lib"] == "FlagGems-vllm"
        assert result["match_type"] == "path"

    def test_sglang_path(self):
        result = parse_reference_url("sglang/python/sglang/srt/layers/fused_moe.py")
        assert result["operator_lib"] == "FlagGems-sglang"
        assert result["match_type"] == "path"

    def test_pytorch_aten_path(self):
        result = parse_reference_url("pytorch/aten/src/ATen/native/Softmax.cpp")
        assert result["operator_lib"] == "FlagGems"

    def test_torch_prefix_path(self):
        result = parse_reference_url("torch/nn/functional.py")
        assert result["operator_lib"] == "FlagGems"

    # ── Edge cases ───────────────────────────────────────────────

    def test_empty_string(self):
        result = parse_reference_url("")
        assert result["operator_lib"] is None
        assert result["match_type"] is None

    def test_none_like_string(self):
        result = parse_reference_url("   ")
        assert result["operator_lib"] is None

    def test_random_text_no_match(self):
        result = parse_reference_url("some random description without keywords")
        assert result["operator_lib"] is None

    def test_keyword_in_text(self):
        result = parse_reference_url("This operator is from vllm attention module")
        assert result["operator_lib"] == "FlagGems-vllm"

    # ── Priority ordering ────────────────────────────────────────

    def test_vllm_keyword_priority_over_torch(self):
        # "vllm" path prefix should match before "torch" keyword
        result = parse_reference_url("vllm/torch_utils/helper.py")
        assert result["operator_lib"] == "FlagGems-vllm"
