"""Identify tools: parse_reference_url.

Determines operator_lib from a reference_impl string using
URL parsing and path pattern matching — no LLM involved.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse


# Mapping: URL host/path keywords → operator_lib
_URL_REPO_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"github\.com/pytorch/pytorch", re.I), "FlagGems"),
    (re.compile(r"github\.com/[^/]+/torch_npu", re.I), "FlagGems"),
    (re.compile(r"github\.com/vllm-project/vllm", re.I), "FlagGems-vllm"),
    (re.compile(r"github\.com/sgl-project/sglang", re.I), "FlagGems-sglang"),
]

# Mapping: file path prefixes/keywords → operator_lib
_PATH_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Explicit repo prefix paths (e.g. "vllm/vllm/...")
    (re.compile(r"^vllm/", re.I), "FlagGems-vllm"),
    (re.compile(r"^sglang/", re.I), "FlagGems-sglang"),
    (re.compile(r"^pytorch/", re.I), "FlagGems"),
    (re.compile(r"^torch_npu/", re.I), "FlagGems"),
    # aten kernel paths
    (re.compile(r"aten/", re.I), "FlagGems"),
    (re.compile(r"torch/", re.I), "FlagGems"),
]


def parse_reference_url(reference_impl: str) -> dict[str, Any]:
    """Parse reference_impl field via URL parsing + path pattern matching.

    This is the first-pass rule-based identification that runs without
    calling an LLM.  It handles:
    - Full GitHub URLs (https://github.com/pytorch/pytorch/blob/...)
    - Relative file paths (vllm/vllm/model_executor/layers/activation.py)
    - Short references containing repo keywords

    Args:
        reference_impl: The reference implementation string from the requirement table.

    Returns:
        {
            "operator_lib": str | None,  # FlagGems / FlagGems-vllm / FlagGems-sglang / None
            "repo_url": str | None,      # extracted repo URL if any
            "match_type": str | None,    # "url" | "path" | None
        }
    """
    if not reference_impl or not reference_impl.strip():
        return {"operator_lib": None, "repo_url": None, "match_type": None}

    text = reference_impl.strip()

    # ── Try URL matching first ───────────────────────────────────
    parsed = urlparse(text)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        full_url = text
        for pattern, lib in _URL_REPO_PATTERNS:
            if pattern.search(full_url):
                return {
                    "operator_lib": lib,
                    "repo_url": full_url,
                    "match_type": "url",
                }
        # URL recognized but repo not in our map.
        return {"operator_lib": None, "repo_url": full_url, "match_type": None}

    # ── Try path pattern matching ────────────────────────────────
    # Normalize backslashes.
    norm = text.replace("\\", "/")
    for pattern, lib in _PATH_PATTERNS:
        if pattern.search(norm):
            return {
                "operator_lib": lib,
                "repo_url": None,
                "match_type": "path",
            }

    # ── Keyword fallback ─────────────────────────────────────────
    lower = norm.lower()
    keyword_map = [
        ("vllm", "FlagGems-vllm"),
        ("sglang", "FlagGems-sglang"),
        ("torch_npu", "FlagGems"),
        ("aten", "FlagGems"),
        ("pytorch", "FlagGems"),
        ("torch", "FlagGems"),
    ]
    for kw, lib in keyword_map:
        if kw in lower:
            return {"operator_lib": lib, "repo_url": None, "match_type": "path"}

    return {"operator_lib": None, "repo_url": None, "match_type": None}
