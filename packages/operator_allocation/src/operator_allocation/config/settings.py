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

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: int = 120
    max_retries: int = 3
    structured_output_mode: str = "openai"  # "openai" | "claude" | "fallback"


class RepoSettings(BaseModel):
    """Local repository base directory."""

    base_dir: str = ""


class ExcelSettings(BaseModel):
    """Excel file paths."""

    dev_table_path: str = ""
    output_dir: str = "output"


class AgentSettings(BaseModel):
    """Agent loop parameters."""

    identify_max_turns: int = 20
    process_max_turns: int = 30
    max_retries: int = 10
    max_tool_result_chars: int = 8000
    context_compress_threshold: int = 30000
    skip_grep_repos: list[str] = Field(default_factory=lambda: ["pytorch"])
    """Repo directory names that skip batch grep verification.

    Grep is skipped when the repo is too large or when URL-pattern matching
    is already reliable (e.g. pytorch/ATen ops). Add a repo name here when
    definition search would be slow or unnecessary for that codebase.
    """


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

        # Validate required fields — fail fast on missing config.
        missing = []
        if not settings.llm.base_url:
            missing.append("llm.base_url")
        if not settings.llm.api_key:
            missing.append("llm.api_key (or env LLM_API_KEY)")
        if not settings.llm.model:
            missing.append("llm.model")
        if missing:
            raise ValueError(
                f"Missing required config: {', '.join(missing)}. "
                f"Set them in {config_path} or via environment variables."
            )

        # Validate repo paths — fail fast so path errors surface at startup,
        # not deep inside a ReAct loop after spending LLM tokens.
        repo_errors = []
        if settings.repos.base_dir:
            base = Path(settings.repos.base_dir)
            if not base.exists():
                repo_errors.append(f"repos.base_dir does not exist: {base}")
            else:
                for lib, repo_dir in settings.lib_repo_mapping.items():
                    repo_path = base / repo_dir
                    if not repo_path.exists():
                        repo_errors.append(
                            f"repo directory for lib '{lib}' does not exist: {repo_path}"
                        )
        elif settings.lib_repo_mapping:
            repo_errors.append(
                "repos.base_dir is empty but lib_repo_mapping is configured"
            )
        if repo_errors:
            raise ValueError(
                "Invalid repo config:\n" + "\n".join(f"  - {e}" for e in repo_errors)
            )

        # Validate no_dev_filter resource files.
        nodev_errors = []
        for lib_name, lib_settings in settings.no_dev_filter.items():
            if lib_name not in settings.lib_repo_mapping:
                nodev_errors.append(
                    f"no_dev_filter lib '{lib_name}' is not in lib_repo_mapping"
                )
            for attr, label in [
                ("known_list_path", "known_list_path"),
                ("prompt_path", "prompt_path"),
            ]:
                file_path = getattr(lib_settings, attr, "")
                if file_path and not Path(file_path).exists():
                    nodev_errors.append(
                        f"no_dev_filter.{lib_name}.{label} does not exist: {file_path}"
                    )
        if nodev_errors:
            raise ValueError(
                "Invalid no_dev_filter config:\n"
                + "\n".join(f"  - {e}" for e in nodev_errors)
            )

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
        """Libs whose repo is in agent.skip_grep_repos (skips batch grep verification).

        Derived from ``agent.skip_grep_repos`` config — set that list to control
        which repos bypass grep. Defaults to ``["pytorch"]`` (ATen ops are
        identified by URL pattern and don't need definition search).
        """
        skip = set(self.agent.skip_grep_repos)
        return {lib for lib, repo in self.lib_repo_mapping.items() if repo in skip}

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
