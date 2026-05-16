"""Data models for SARIF findings and triage results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Location:
    """A source code location from SARIF."""

    file_path: str
    start_line: int
    end_line: int
    start_column: int | None = None
    end_column: int | None = None

    def __str__(self) -> str:
        return f"{self.file_path}:{self.start_line}"


@dataclass
class CodeFlowStep:
    """A single step in a SARIF codeFlow."""

    location: Location
    message: str = ""


@dataclass
class Finding:
    """A single vulnerability finding extracted from a SARIF report."""

    rule_id: str
    message: str
    severity: str  # error / warning / note
    level: str  # SARIF level
    location: Location
    code_flows: list[list[CodeFlowStep]] = field(default_factory=list)
    related_locations: list[Location] = field(default_factory=list)
    properties: dict = field(default_factory=dict)

    # Original SARIF tool info
    tool_name: str = ""
    tool_version: str = ""

    @property
    def file_path(self) -> str:
        return self.location.file_path

    @property
    def start_line(self) -> int:
        return self.location.start_line

    @property
    def end_line(self) -> int:
        return self.location.end_line


@dataclass
class TriageVerdict:
    """LLM triage verdict for a single finding."""

    verdict: Literal["true_positive", "false_positive", "uncertain"]
    confidence: float  # 0.0 — 1.0
    reasoning: str


@dataclass
class TriageResult:
    """Complete triage result for a single finding."""

    finding: Finding
    verdict: TriageVerdict
    code_snippet: str
    codeql_context: str | None = None
    llm_model: str = ""
    llm_tokens_used: int = 0
    error: str | None = None


@dataclass
class TriageReport:
    """Aggregated triage report for all findings."""

    sarif_file: str
    source_path: str
    llm_model: str
    results: list[TriageResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def true_positives(self) -> int:
        return sum(1 for r in self.results if r.verdict.verdict == "true_positive")

    @property
    def false_positives(self) -> int:
        return sum(1 for r in self.results if r.verdict.verdict == "false_positive")

    @property
    def uncertain(self) -> int:
        return sum(1 for r in self.results if r.verdict.verdict == "uncertain")

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.error is not None)
