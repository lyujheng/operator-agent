"""No-dev filter models: NoDevFilterProfile, NoDevJudgment, NoDevBatchResult.

Configuration and structured output schemas for the no_dev filtering module,
which determines operators that do not need FlagGems (Triton GPU kernel) development.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Configuration Model ────────────────────────────────────────

# Valid no_dev categories (LLM must pick from this set).
DEFAULT_VALID_CATEGORIES: set[str] = {
    "view_shape",
    "attribute_query",
    "platform_specific",
    "copy_variant",
    "memory_management",
    "sparse_construction",
    "autograd_control",
    "symbolic_validation",
    "test_internal",
}


class NoDevFilterProfile(BaseModel):
    """Per-lib no_dev filter configuration.

    Each operator library can have its own profile with separate known lists,
    prompts, and category definitions. Adding a new lib = create files +
    register a NoDevFilterProfile instance.
    """

    lib_name: str                           # e.g. "FlagGems"
    known_list_path: str                    # confirmed no_dev operator list (human-curated)
    pending_list_path: str                  # auto-accumulated from LLM judgments
    prompt_path: str                        # Tier 2 LLM classification system prompt
    llm_batch_size: int = 8                 # Tier 2 batch size
    confidence_threshold: str = "high"      # below this → needs_dev
    valid_categories: set[str] = Field(default_factory=lambda: set(DEFAULT_VALID_CATEGORIES))


# ── Structured Output Schemas ──────────────────────────────────

class NoDevJudgment(BaseModel):
    """Single operator no_dev judgment from LLM."""

    operator_name: str
    verdict: Literal["no_dev", "needs_dev"]
    category: str | None = None             # no_dev: one of the lib profile's valid_categories; needs_dev: None
    confidence: Literal["high", "medium", "low"]
    reasoning: str                          # judgment rationale


class NoDevBatchResult(BaseModel):
    """Batch LLM classification result."""

    judgments: list[NoDevJudgment]
