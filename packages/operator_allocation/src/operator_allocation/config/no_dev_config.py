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
    known_list_path="repo/torch_no_dev_ops.md",
    pending_list_path="repo/torch_no_dev_ops_pending.md",
    prompt_path="repo/prompts/torch_no_dev_prompt.md",
    llm_batch_size=8,
    confidence_threshold="high",
)

# ── Profile Registry ───────────────────────────────────────────
# Maps operator_lib name -> NoDevFilterProfile.
# Libs not in this registry skip filtering (all operators pass to dispatch).

NO_DEV_PROFILES: dict[str, NoDevFilterProfile] = {
    "FlagGems": FLAGGEMS_PROFILE,
}
