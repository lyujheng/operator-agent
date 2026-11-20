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

    base_url: str = "https://zyapi.xmsxb.com/v1"
    api_key: str = "your-api-key"
    model: str = "mco-4"
    timeout: int = 120
    max_retries: int = 3


class RepoSettings(BaseModel):
    """Local repository paths."""

    base_dir: str = "/data/repos"

    @property
    def pytorch_path(self) -> str:
        return str(Path(self.base_dir) / "pytorch")

    @property
    def torch_npu_path(self) -> str:
        return str(Path(self.base_dir) / "torch_npu")

    @property
    def vllm_path(self) -> str:
        return str(Path(self.base_dir) / "vllm")

    @property
    def sglang_path(self) -> str:
        return str(Path(self.base_dir) / "sglang")


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


class AppSettings(BaseSettings):
    """Top-level settings loaded from a TOML config file.

    Usage::

        settings = AppSettings.load("config.toml")
    """

    llm: LLMSettings = Field(default_factory=LLMSettings)
    repos: RepoSettings = Field(default_factory=RepoSettings)
    excel: ExcelSettings = Field(default_factory=ExcelSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)

    model_config = {
        "toml_file": "config.toml",
    }

    @classmethod
    def load(cls, config_path: str | Path) -> AppSettings:
        """Load settings from a specific TOML file path."""
        import tomllib

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(**data)

    @property
    def repo_lib_mapping(self) -> dict[str, str]:
        """Map operator_lib names to local repo paths."""
        return {
            "FlagGems": self.repos.pytorch_path,
            "FlagGems-vllm": self.repos.vllm_path,
            "FlagGems-sglang": self.repos.sglang_path,
        }
