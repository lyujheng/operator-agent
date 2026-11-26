"""Application settings - pydantic BaseSettings loading from config.toml.

All runtime parameters are centralized here and validated at startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMSettings(BaseModel):
    """LLM API connection parameters."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = "your-api-key"
    model: str = "mco-4"
    timeout: int = 120
    max_retries: int = 3
    structured_output_mode: str = "openai"  # "openai" | "claude" | "fallback"


class RepoSettings(BaseModel):
    """Local repository base directory."""

    base_dir: str = "/data/repos"


class ExcelSettings(BaseModel):
    """Excel file paths."""

    dev_table_path: str = "/data/dev_tables/dev_table.xlsx"
    output_dir: str = "/data/output"


class AgentSettings(BaseModel):
    """Agent loop parameters."""

    identify_max_turns: int = 20
    process_max_turns: int = 30
    max_retries: int = 10
    max_tool_result_chars: int = 8000
    context_compress_threshold: int = 30000


class NoDevFilterLibSettings(BaseModel):
    """Per-lib no_dev filter settings from config.toml."""

    known_list_path: str = ""
    pending_list_path: str = ""
    prompt_path: str = ""
    llm_batch_size: int = 8
    confidence_threshold: str = "high"


class AppSettings(BaseSettings):
    """Top-level settings loaded from a TOML config file.

    Usage::

        settings = AppSettings.load("config.toml")
    """

    llm: LLMSettings = Field(default_factory=LLMSettings)
    repos: RepoSettings = Field(default_factory=RepoSettings)
    excel: ExcelSettings = Field(default_factory=ExcelSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    no_dev_filter: dict[str, NoDevFilterLibSettings] = Field(default_factory=dict)
    lib_repo_mapping: dict[str, str] = Field(default_factory=dict)
    """Maps dev table group (lib/sheet name) → repo directory name.

    Loaded from ``[lib_repo_mapping]`` in config.toml.
    Example: ``{"FlagGems": "pytorch", "FlagGems-vllm": "vllm"}``.
    """

    model_config = {}  # loaded via classmethod, not pydantic-settings sources

    @classmethod
    def load(cls, config_path: str | Path) -> AppSettings:
        """Load settings from a specific TOML file path.

        After loading the TOML, applies environment variable overrides:
          - ``LLM_API_KEY`` → ``llm.api_key``
          - ``LLM_BASE_URL`` → ``llm.base_url``
          - ``LLM_MODEL`` → ``llm.model``
        """
        import os
        import tomllib

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        settings = cls(**data)

        # Environment variable overrides.
        if env_key := os.environ.get("LLM_API_KEY"):
            settings.llm.api_key = env_key
        if env_url := os.environ.get("LLM_BASE_URL"):
            settings.llm.base_url = env_url
        if env_model := os.environ.get("LLM_MODEL"):
            settings.llm.model = env_model

        return settings

    # ── Computed properties derived from lib_repo_mapping ────────

    @property
    def valid_libs(self) -> set[str]:
        """All valid operator_lib values: mapped libs + "unrecognized"."""
        return set(self.lib_repo_mapping.keys()) | {"unrecognized"}

    @property
    def repo_libs(self) -> set[str]:
        """Libs that map to a source code repo (excludes "unrecognized")."""
        return set(self.lib_repo_mapping.keys())

    @property
    def skip_grep_libs(self) -> set[str]:
        """Libs whose repo is too large for per-operator grep verification.

        Currently: libs mapping to "pytorch" (ATen ops identified by URL
        pattern, no grep needed).
        """
        return {lib for lib, repo in self.lib_repo_mapping.items() if repo == "pytorch"}

    @property
    def all_repo_names(self) -> set[str]:
        """All unique repo directory names from the mapping."""
        return set(self.lib_repo_mapping.values())

    @property
    def repo_lib_mapping(self) -> dict[str, str]:
        """Map lib names to absolute repo paths (lib → absolute path)."""
        return {
            lib: str(Path(self.repos.base_dir) / repo_dir)
            for lib, repo_dir in self.lib_repo_mapping.items()
        }

    def get_repo_path(self, lib: str) -> str:
        """Get absolute repo path for a lib. Returns empty string if unmapped."""
        repo_dir = self.lib_repo_mapping.get(lib, "")
        if not repo_dir:
            return ""
        return str(Path(self.repos.base_dir) / repo_dir)

    def get_repo_name(self, lib: str) -> str:
        """Get repo directory name for a lib. Returns empty string if unmapped."""
        return self.lib_repo_mapping.get(lib, "")
