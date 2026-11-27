"""No-dev filter configuration: per-lib profiles and registry.

Profiles configure the no_dev filtering behavior for each operator library.
To add a new lib: create its known_list, pending_list, and prompt files,
then register a NoDevFilterProfile here.
"""

from __future__ import annotations

from operator_allocation.models.no_dev import NoDevFilterProfile

# ── FlagGems Profile ───────────────────────────────────────────

FLAGGEMS_PROFILE = NoDevFilterProfile(
    lib_name="FlagGems",
    known_list_path="resources/no_dev/torch_no_dev_ops.md",
    pending_list_path="resources/no_dev/torch_no_dev_ops_pending.md",
    prompt_path="resources/no_dev/prompts/torch_no_dev_prompt.md",
    llm_batch_size=8,
    confidence_threshold="high",
)

# ── FlagGems-vllm Profile ─────────────────────────────────────
# vllm wrapper operators: compile_wrapper is the only valid no_dev
# category — intentionally separate from FlagGems' DEFAULT_VALID_CATEGORIES.

FLAGGEMS_VLLM_PROFILE = NoDevFilterProfile(
    lib_name="FlagGems-vllm",
    known_list_path="resources/no_dev/vllm_no_dev_ops.md",
    pending_list_path="resources/no_dev/vllm_no_dev_ops_pending.md",
    prompt_path="resources/no_dev/prompts/vllm_no_dev_prompt.md",
    llm_batch_size=8,
    confidence_threshold="high",
    valid_categories={"compile_wrapper"},
)

# ── Profile Registry ───────────────────────────────────────────
# Maps operator_lib name -> NoDevFilterProfile.
# Libs not in this registry skip filtering (all operators pass to dispatch).

NO_DEV_PROFILES: dict[str, NoDevFilterProfile] = {
    "FlagGems": FLAGGEMS_PROFILE,
    "FlagGems-vllm": FLAGGEMS_VLLM_PROFILE,
}
