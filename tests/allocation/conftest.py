"""Shared fixtures for operator_allocation tests."""

from __future__ import annotations

import pytest

from operator_allocation.tools.data_tools import set_valid_libs
from operator_allocation.tools.identify_tools import configure_patterns

# Standard lib/repo mapping used across all allocation tests.
STANDARD_LIB_REPO_MAPPING = {
    "FlagGems": "pytorch",
    "FlagGems-vllm": "vllm",
    "FlagGems-sglang": "sglang",
}

STANDARD_VALID_LIBS = set(STANDARD_LIB_REPO_MAPPING.keys()) | {"unrecognized"}


@pytest.fixture(autouse=True)
def configure_identify_patterns():
    """Configure identify_tools patterns from the standard lib_repo_mapping."""
    configure_patterns(STANDARD_LIB_REPO_MAPPING)


@pytest.fixture(autouse=True)
def configure_valid_libs():
    """Inject valid_libs into data_tools so check_result fallback is populated."""
    set_valid_libs(STANDARD_VALID_LIBS)
