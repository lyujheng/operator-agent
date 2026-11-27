"""Identify tools: parse_reference_url.

Determines operator_lib from a reference_impl string using
URL parsing and path pattern matching — no LLM involved.

Call ``configure_patterns(lib_repo_mapping)`` once at startup to derive
patterns from the config instead of relying on the built-in defaults.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse


# Mapping: URL host/path keywords → operator_lib
# Initialized empty; call configure_patterns(lib_repo_mapping) before use.
_URL_REPO_PATTERNS: list[tuple[re.Pattern[str], str]] = []

# Mapping: file path prefixes/keywords → operator_lib
_PATH_PATTERNS: list[tuple[re.Pattern[str], str]] = []

# Keyword fallback: (keyword, operator_lib) in priority order.
_KEYWORD_MAP: list[tuple[str, str]] = []

# Tracks whether configure_patterns() has been called.
_patterns_configured: bool = False


def configure_patterns(lib_repo_mapping: dict[str, str]) -> None:
    """Rebuild URL/path/keyword patterns from ``lib_repo_mapping``.

    Called once at startup by ``AgentOrchestrator._identify_operators`` so
    that adding a new entry to ``[lib_repo_mapping]`` in config.toml is
    sufficient — no code changes required here.

    Pattern generation rules (per lib → repo_dir pair):
    - URL: ``github.com/<any-org>/<repo_dir>``
    - Path prefix: ``^<repo_dir>/``
    - Keyword: ``<repo_dir>`` substring match

    PyTorch aliases (aten/, torch/, torch_npu) are added automatically
    when the mapped repo_dir is ``"pytorch"``.

    Args:
        lib_repo_mapping: ``{lib_name: repo_dir}`` from ``AppSettings``,
            e.g. ``{"FlagGems": "pytorch", "FlagGems-vllm": "vllm"}``.
    """
    global _URL_REPO_PATTERNS, _PATH_PATTERNS, _KEYWORD_MAP, _patterns_configured

    url_patterns: list[tuple[re.Pattern[str], str]] = []
    path_patterns: list[tuple[re.Pattern[str], str]] = []
    keyword_map: list[tuple[str, str]] = []

    for lib, repo_dir in lib_repo_mapping.items():
        rd = repo_dir.lower()
        # GitHub URL: github.com/<any-org>/<repo_dir>
        url_patterns.append(
            (re.compile(rf"github\.com/[^/]+/{re.escape(rd)}", re.I), lib)
        )
        # Path prefix
        path_patterns.append(
            (re.compile(rf"^{re.escape(rd)}/", re.I), lib)
        )
        # Keyword
        keyword_map.append((rd, lib))

        # PyTorch-specific aliases
        if rd == "pytorch":
            url_patterns.append(
                (re.compile(r"github\.com/[^/]+/torch_npu", re.I), lib)
            )
            path_patterns.append((re.compile(r"^torch_npu/", re.I), lib))
            path_patterns.append((re.compile(r"aten/", re.I), lib))
            path_patterns.append((re.compile(r"torch/", re.I), lib))
            keyword_map.extend([("torch_npu", lib), ("aten", lib), ("torch", lib)])

    _URL_REPO_PATTERNS = url_patterns
    _PATH_PATTERNS = path_patterns
    _KEYWORD_MAP = keyword_map
    _patterns_configured = True


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

    if not _patterns_configured:
        raise RuntimeError(
            "identify_tools patterns not configured. "
            "Call configure_patterns(lib_repo_mapping) before parse_reference_url()."
        )

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
    for kw, lib in _KEYWORD_MAP:
        if kw in lower:
            return {"operator_lib": lib, "repo_url": None, "match_type": "path"}

    return {"operator_lib": None, "repo_url": None, "match_type": None}
