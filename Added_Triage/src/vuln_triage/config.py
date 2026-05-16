"""Application configuration with pydantic-settings."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class CodeQLConfig(BaseModel):
    """Configuration for CodeQL analysis."""

    codeql_path: str = "codeql"
    search_path: str = ""
    language: str = "python"
    db_timeout: int = 600
    query_timeout: int = 300


class LLMConfig(BaseModel):
    model: str = "openrouter/anthropic/claude-sonnet-4-6"
    temperature: float = 0.1
    max_tokens: int = 2048
    max_concurrent: int = 5


class ExtractionConfig(BaseModel):
    context_lines: int = 30
    max_snippet_lines: int = 100


class PipelineConfig(BaseModel):
    max_concurrent_findings: int = 10
    fallback_to_ast: bool = True


class AppConfig(BaseModel):
    codeql: CodeQLConfig = Field(default_factory=CodeQLConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load configuration from YAML file, falling back to defaults."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return AppConfig(**data)
    return AppConfig()
