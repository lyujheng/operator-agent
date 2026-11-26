"""No-dev filter: determine which operators do not need FlagGems development.

Two-tier filtering:
  Tier 1 — exact match against known + pending lists
  Tier 2 — LLM batch classification with structured output

This module runs behind MCP (NO_DEV_SERVER).  The orchestrator injects
context via ``set_filter_context()`` before calling the tool.  When called
through MCP, ``filter_no_dev_operators()`` reads from module-level state
and internally creates an LLMClient for Tier 2 classification.

Backward compatibility: passing explicit ``operators`` / ``profile`` /
``llm_client`` arguments still works (used by unit tests).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from operator_allocation.models.no_dev import (
    NoDevBatchResult,
    NoDevFilterProfile,
    NoDevJudgment,
)

logger = logging.getLogger(__name__)


# ── Module-Level State (injected by MCP server on startup) ────

_filter_operators: list[dict[str, Any]] = []
_filter_profile: NoDevFilterProfile | None = None
_llm_settings: dict[str, Any] = {}


def set_filter_context(
    operators: list[dict[str, Any]],
    profile_dict: dict[str, Any],
    llm_settings: dict[str, Any],
) -> None:
    """Inject filter context into module.  Called by MCP server during init."""
    global _filter_operators, _filter_profile, _llm_settings
    _filter_operators = operators
    _filter_profile = NoDevFilterProfile(**profile_dict)
    _llm_settings = llm_settings
    logger.info(
        "Filter context set: %d operators, lib=%s",
        len(operators),
        _filter_profile.lib_name,
    )


def get_filter_context() -> tuple[list[dict[str, Any]], NoDevFilterProfile | None, dict[str, Any]]:
    """Return current filter context (operators, profile, llm_settings)."""
    return _filter_operators, _filter_profile, _llm_settings

# Regex for markdown table rows: | # | operator_name |
_TABLE_ROW_RE = re.compile(r"^\|\s*\d+\s*\|\s*(\S+)\s*\|")

# Regex for category headers: ## 类别名（N）
_CATEGORY_HEADER_RE = re.compile(r"^##\s+(.+?)(?:[（(]\d+[)）])?\s*$")


# ── Known List Parsing ─────────────────────────────────────────

def load_no_dev_known_list(file_path: str) -> dict[str, str]:
    """Parse no_dev known list markdown file (category-grouped tables).

    File format: markdown with ``## Category`` headers followed by
    ``| # | operator_name |`` table rows.

    Args:
        file_path: Path to the markdown file.

    Returns:
        ``{operator_name: category_name}`` mapping.
    """
    path = Path(file_path)
    if not path.exists():
        logger.debug("Known list file not found: %s", file_path)
        return {}

    result: dict[str, str] = {}
    current_category = "unknown"

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read known list %s: %s", file_path, exc)
        return {}

    for line in text.splitlines():
        line = line.strip()

        # Match category header.
        header_match = _CATEGORY_HEADER_RE.match(line)
        if header_match:
            current_category = _normalize_category(header_match.group(1))
            continue

        # Match table row.
        row_match = _TABLE_ROW_RE.match(line)
        if row_match:
            op_name = row_match.group(1).strip()
            if op_name and op_name != "算子名":  # skip header row
                result[op_name] = current_category

    logger.debug("Loaded %d operators from %s", len(result), file_path)
    return result


def _normalize_category(raw: str) -> str:
    """Map Chinese category headers to canonical category slugs."""
    mapping = {
        # FlagGems (pytorch) categories
        "视图/形状变换": "view_shape",
        "视图": "view_shape",
        "形状变换": "view_shape",
        "属性查询": "attribute_query",
        "平台专属实现": "platform_specific",
        "平台专属": "platform_specific",
        "Copy 变体": "copy_variant",
        "copy 变体": "copy_variant",
        "copy变体": "copy_variant",
        "测试/内部工具": "test_internal",
        "测试": "test_internal",
        "内部工具": "test_internal",
        "符号/断言": "symbolic_validation",
        "符号": "symbolic_validation",
        "断言": "symbolic_validation",
        "稀疏张量构造": "sparse_construction",
        "稀疏": "sparse_construction",
        "内存管理": "memory_management",
        "cuFFT 运行时管理": "memory_management",
        "cuFFT": "memory_management",
        "Autograd 控制": "autograd_control",
        "autograd 控制": "autograd_control",
        "Nested Tensor 视图": "view_shape",
        # FlagGems-vllm categories
        "compile_wrapper": "compile_wrapper",
        "编译包装": "compile_wrapper",
        "torch.compile wrapper": "compile_wrapper",
    }
    for key, slug in mapping.items():
        if key in raw:
            return slug
    return "unknown"


# ── Main Filter Entry ──────────────────────────────────────────

async def filter_no_dev_operators(
    operators: list[dict[str, Any]] | None = None,
    profile: NoDevFilterProfile | None = None,
    llm_client: Any | None = None,
) -> dict[str, Any]:
    """Main entry for no_dev filtering. Tier 1 exact match + Tier 2 LLM classify.

    Modifies operators in-place: sets ``skip_dev`` (bool) and
    ``skip_reason`` (str | None) on each operator whose
    ``operator_lib`` matches ``profile.lib_name``.

    When called through MCP (no arguments), reads operators/profile from
    module-level state and creates an LLMClient internally from
    ``_llm_settings``.  Explicit arguments are still accepted for
    backward compatibility (unit tests).

    Returns:
        ``{"total": int, "tier1_no_dev": int, "tier2_no_dev": int,
           "pass_to_dispatch": int, "operators": list}``
        The ``operators`` key contains modified operator dicts so the
        MCP caller can map results back to its own state.
    """
    # Resolve from module state when called without arguments (MCP mode).
    if operators is None:
        operators = _filter_operators
    if profile is None:
        profile = _filter_profile
    if profile is None:
        raise ValueError("No profile: pass explicitly or call set_filter_context first")

    # Build an LLMClient from module-level settings when no client given.
    _own_llm_client = False
    if llm_client is None and _llm_settings:
        from agent_framework.llm.client import LLMClient

        llm_client = LLMClient(
            base_url=_llm_settings.get("base_url", ""),
            api_key=_llm_settings.get("api_key", ""),
            model=_llm_settings.get("model", "gpt-4"),
            timeout=_llm_settings.get("timeout", 120),
            max_retries=_llm_settings.get("max_retries", 3),
            structured_output_mode=_llm_settings.get("structured_output_mode", "openai"),
        )
        _own_llm_client = True

    # Select operators belonging to this lib.
    target_ops = [
        op for op in operators
        if op.get("operator_lib") == profile.lib_name
    ]
    total = len(target_ops)

    if total == 0:
        return {"total": 0, "tier1_no_dev": 0, "tier2_no_dev": 0, "pass_to_dispatch": 0, "operators": []}

    # ── Load known lists ──
    known_map = load_no_dev_known_list(profile.known_list_path)
    pending_map = load_no_dev_known_list(profile.pending_list_path)
    full_map = {**known_map, **pending_map}

    logger.info(
        "No-dev filter for %s: %d operators, %d known + %d pending entries",
        profile.lib_name, total, len(known_map), len(pending_map),
    )

    # ── Tier 1: Exact match ──
    tier1_matched: list[dict[str, Any]] = []
    tier1_remaining: list[dict[str, Any]] = []

    for op in target_ops:
        op_name = op.get("operator_name", "")
        if op_name in full_map:
            op["skip_dev"] = True
            op["skip_reason"] = f"exact_match:{full_map[op_name]}"
            tier1_matched.append(op)
        else:
            tier1_remaining.append(op)

    tier1_no_dev = len(tier1_matched)
    logger.info("Tier 1 exact match: %d no_dev, %d remaining", tier1_no_dev, len(tier1_remaining))

    # ── Tier 2: LLM batch classification ──
    tier2_no_dev = 0
    new_pending: list[dict[str, Any]] = []

    if tier1_remaining:
        # Split into batches.
        batches = _chunk(tier1_remaining, profile.llm_batch_size)
        for batch_idx, batch in enumerate(batches):
            try:
                batch_result = await llm_classify_batch(batch, profile, llm_client)
                tier2_count = _apply_llm_judgments(
                    batch, batch_result, profile, new_pending,
                )
                tier2_no_dev += tier2_count
            except Exception as exc:
                # LLM failure: conservative fallback — all needs_dev.
                logger.warning(
                    "LLM batch %d/%d failed: %s, defaulting to needs_dev",
                    batch_idx + 1, len(batches), exc,
                )
                for op in batch:
                    op["skip_dev"] = False
                    op["skip_reason"] = None

    logger.info("Tier 2 LLM classify: %d no_dev", tier2_no_dev)

    # ── Pending list auto-accumulation ──
    if new_pending:
        try:
            appended = append_to_pending_list(profile.pending_list_path, new_pending)
            logger.info("Appended %d entries to pending list", appended)
        except Exception as exc:
            logger.warning("Failed to append to pending list: %s", exc)

    # Close internally-created LLMClient.
    if _own_llm_client and llm_client is not None:
        try:
            await llm_client.close()
        except Exception:
            pass

    pass_to_dispatch = total - tier1_no_dev - tier2_no_dev
    return {
        "total": total,
        "tier1_no_dev": tier1_no_dev,
        "tier2_no_dev": tier2_no_dev,
        "pass_to_dispatch": pass_to_dispatch,
        "operators": target_ops,
    }


def _apply_llm_judgments(
    batch: list[dict[str, Any]],
    batch_result: NoDevBatchResult,
    profile: NoDevFilterProfile,
    new_pending: list[dict[str, Any]],
) -> int:
    """Apply LLM judgments to a batch of operators.

    Returns the count of operators marked as no_dev.
    """
    # Build name->judgment map for quick lookup.
    judgment_map: dict[str, NoDevJudgment] = {
        j.operator_name: j for j in batch_result.judgments
    }

    no_dev_count = 0
    for op in batch:
        op_name = op.get("operator_name", "")
        judgment = judgment_map.get(op_name)

        if judgment is None:
            # LLM didn't return judgment for this operator — conservative.
            op["skip_dev"] = False
            op["skip_reason"] = None
            continue

        if (
            judgment.verdict == "no_dev"
            and judgment.confidence == profile.confidence_threshold
            and judgment.category in profile.valid_categories
        ):
            op["skip_dev"] = True
            op["skip_reason"] = f"llm:{judgment.category}"
            new_pending.append({
                "operator_name": op_name,
                "category": judgment.category,
                "reasoning": judgment.reasoning,
            })
            no_dev_count += 1
        else:
            op["skip_dev"] = False
            op["skip_reason"] = None

    return no_dev_count


# ── LLM Batch Classification ──────────────────────────────────

async def llm_classify_batch(
    operators: list[dict[str, Any]],
    profile: NoDevFilterProfile,
    llm_client: Any,
) -> NoDevBatchResult:
    """Classify a batch of operators via LLM structured output.

    Uses ``profile.prompt_path`` as system prompt, returns
    ``NoDevBatchResult``. Retries once on schema validation failure,
    falls back to all needs_dev on persistent failure.

    Args:
        operators: Batch of operator dicts.
        profile: No-dev filter profile with prompt path.
        llm_client: LLMClient instance.

    Returns:
        ``NoDevBatchResult`` with one judgment per input operator.
    """
    system_prompt = _read_prompt_file(profile.prompt_path)
    user_message = _format_batch_for_llm(operators, profile.lib_name)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # Try structured output first, retry once on failure.
    for attempt in range(2):
        try:
            result = await llm_client.get_structured_output(messages, NoDevBatchResult)
            return result
        except Exception as exc:
            if attempt == 0:
                logger.warning("LLM structured output attempt 1 failed: %s, retrying", exc)
            else:
                logger.error("LLM structured output attempt 2 failed: %s", exc)
                raise

    # Should not reach here, but satisfy type checker.
    raise RuntimeError("LLM classification failed after retries")  # pragma: no cover


def _read_prompt_file(prompt_path: str) -> str:
    """Read the system prompt from a markdown file."""
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return path.read_text(encoding="utf-8")


def _format_batch_for_llm(operators: list[dict[str, Any]], lib_name: str = "FlagGems") -> str:
    """Format a batch of operators into LLM user message."""
    lines = [f"请判断以下算子是否需要 {lib_name} 开发：", ""]
    for i, op in enumerate(operators, 1):
        op_name = op.get("operator_name", "")
        ref = op.get("reference_impl", "")
        if ref:
            lines.append(f"{i}. {op_name} (参考: {ref})")
        else:
            lines.append(f"{i}. {op_name}")
    return "\n".join(lines)


# ── Pending List Append ────────────────────────────────────────

def append_to_pending_list(file_path: str, new_entries: list[dict[str, Any]]) -> int:
    """Append LLM-judged no_dev operators to pending list file.

    Deduplicates by operator_name against existing entries.

    Args:
        file_path: Path to the pending list markdown file.
        new_entries: List of ``{"operator_name": str, "category": str, "reasoning": str}``.

    Returns:
        Number of actually appended entries.
    """
    path = Path(file_path)

    # Load existing names for dedup.
    existing_names: set[str] = set()
    if path.exists():
        existing_map = load_no_dev_known_list(file_path)
        existing_names = set(existing_map.keys())

    # Filter out duplicates.
    to_append = [
        entry for entry in new_entries
        if entry.get("operator_name", "") not in existing_names
    ]

    if not to_append:
        return 0

    # Group by category for organized output.
    by_category: dict[str, list[dict[str, Any]]] = {}
    for entry in to_append:
        cat = entry.get("category", "unknown")
        by_category.setdefault(cat, []).append(entry)

    # Append to file.
    lines: list[str] = ["\n"]
    for category, entries in sorted(by_category.items()):
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| # | 算子名 | 判定理由 |")
        lines.append("|---|--------|----------|")
        for i, entry in enumerate(entries, 1):
            name = entry.get("operator_name", "")
            reasoning = entry.get("reasoning", "").replace("|", "\\|")
            lines.append(f"| {i} | {name} | {reasoning} |")
        lines.append("")

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Appended %d new entries to pending list %s", len(to_append), file_path)
    return len(to_append)


# ── Utility ────────────────────────────────────────────────────

def _chunk(items: list, size: int) -> list[list]:
    """Split a list into chunks of the given size."""
    return [items[i:i + size] for i in range(0, len(items), size)]
