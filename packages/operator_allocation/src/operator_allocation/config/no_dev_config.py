"""No-dev filter configuration: per-lib profiles and registry.

Profiles configure the no_dev filtering behavior for each operator library.
To add a new lib: create its known_list, pending_list, and prompt files,
then register a NoDevFilterProfile here.
"""

from __future__ import annotations

from pathlib import Path

from operator_allocation.models.no_dev import NoDevFilterProfile

# Resolve the project root relative to this file so profile paths are
# CWD-independent.  Structure: packages/operator_allocation/src/operator_allocation/config/no_dev_config.py
# → 5 parents up = packages/ → 1 more = project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_NO_DEV_DIR = _PROJECT_ROOT / "resources" / "no_dev"


def _res(relative: str) -> str:
    """Return an absolute path under resources/no_dev/."""
    return str(_NO_DEV_DIR / relative)


# ── FlagGems Profile ───────────────────────────────────────────

FLAGGEMS_PROFILE = NoDevFilterProfile(
    lib_name="FlagGems",
    known_list_path=_res("torch_no_dev_ops.md"),
    pending_list_path=_res("torch_no_dev_ops_pending.md"),
    prompt_path=_res("prompts/torch_no_dev_prompt.md"),
    llm_batch_size=8,
    confidence_threshold="high",
)

# ── FlagGems-vllm Profile ─────────────────────────────────────
# vllm wrapper operators: compile_wrapper is the only valid no_dev
# category — intentionally separate from FlagGems' DEFAULT_VALID_CATEGORIES.

FLAGGEMS_VLLM_PROFILE = NoDevFilterProfile(
    lib_name="FlagGems-vllm",
    known_list_path=_res("vllm_no_dev_ops.md"),
    pending_list_path=_res("vllm_no_dev_ops_pending.md"),
    prompt_path=_res("prompts/vllm_no_dev_prompt.md"),
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
