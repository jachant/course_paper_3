"""High-level CodeQL analyser — extracts graph context for LLM prompts.

For every SARIF finding the analyser runs up to six CodeQL queries and
assembles the results into a single text block that is injected into
the LLM prompt.

Sections
--------
1. TAINT FLOW PATHS   — full source → sink chains
2. SANITIZERS & GUARDS — barrier functions on the data path
3. ENCLOSING SCOPE     — function/method containing the finding
4. CALL GRAPH          — callers + callees of that function
5. SINK CLASSIFICATION — CodeQL Concepts type of the sink
6. SOURCE CLASSIFICATION — type of the data source
"""

from __future__ import annotations

import logging
from pathlib import Path

from vuln_triage.config import CodeQLConfig
from vuln_triage.sarif.models import Finding

from .codeql_client import CodeQLClient, CodeQLError
from .queries import (
    call_graph_query,
    sanitizers_query,
    scope_query,
    sink_classification_query,
    source_classification_query,
    taint_paths_simple_query,
)

logger = logging.getLogger(__name__)


class CodeQLAnalyzer:
    """Extracts CodeQL graph context relevant to vulnerability findings."""

    def __init__(self, source_root: Path, config: CodeQLConfig | None = None):
        config = config or CodeQLConfig()
        self.source_root = source_root
        self.client = CodeQLClient(
            codeql_path=config.codeql_path,
            search_path=config.search_path or None,
        )
        self._db_ready = False
        self._config = config

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def ensure_db(self) -> None:
        """Create the CodeQL database if it hasn't been created yet."""
        if not self._db_ready:
            self.client.create_database(
                self.source_root,
                language=self._config.language,
                timeout=self._config.db_timeout,
            )
            self._db_ready = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, finding: Finding) -> str | None:
        """Run all CodeQL queries for a finding and return formatted context.

        Returns ``None`` if the database cannot be built or all queries
        return empty results.
        """
        try:
            self.ensure_db()
        except CodeQLError as exc:
            logger.error("CodeQL database creation failed: %s", exc)
            return None

        filepath = finding.file_path
        line = finding.start_line

        sections: list[str] = []

        # 1. Taint flow paths (most valuable — run first)
        taint_text = self._run_taint_paths(filepath, line)
        if taint_text:
            sections.append(taint_text)

        # 2. Sanitizers & guards
        sanitizers_text = self._run_sanitizers(filepath, line)
        if sanitizers_text:
            sections.append(sanitizers_text)

        # 3. Enclosing scope
        scope_text = self._run_scope(filepath, line)
        if scope_text:
            sections.append(scope_text)

        # 4. Call graph
        cg_text = self._run_call_graph(filepath, line)
        if cg_text:
            sections.append(cg_text)

        # 5. Sink classification
        sink_text = self._run_sink_classification(filepath, line)
        if sink_text:
            sections.append(sink_text)

        # 6. Source classification
        source_text = self._run_source_classification(filepath, line)
        if source_text:
            sections.append(source_text)

        if not sections:
            return None

        header = f"CodeQL Analysis for {filepath}:{line}"
        return f"{header}\n{'=' * len(header)}\n\n" + "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Per-section runners
    # ------------------------------------------------------------------

    def _safe_query(self, ql: str, label: str) -> list[dict[str, str]]:
        """Run a query, returning [] on any error."""
        try:
            return self.client.run_query_text(ql, timeout=self._config.query_timeout)
        except CodeQLError as exc:
            logger.warning("CodeQL query [%s] failed: %s", label, exc)
            return []

    # -- 1. taint paths ------------------------------------------------

    def _run_taint_paths(self, filepath: str, line: int) -> str | None:
        ql = taint_paths_simple_query(filepath, line)
        rows = self._safe_query(ql, "taint_paths")
        if not rows:
            return None

        lines: list[str] = []
        for i, row in enumerate(rows[:5], 1):
            src = f"{row.get('source_label', '?')} @ {row.get('source_file', '?')}:{row.get('source_line', '?')}"
            snk = f"{row.get('sink_label', '?')} @ {row.get('sink_file', '?')}:{row.get('sink_line', '?')}"
            lines.append(f"Path {i}:")
            lines.append(f"  [source] {src}")
            lines.append(f"  [sink]   {snk}")

        return "=== TAINT FLOW PATHS ===\n" + "\n".join(lines)

    # -- 2. sanitizers -------------------------------------------------

    def _run_sanitizers(self, filepath: str, line: int) -> str | None:
        ql = sanitizers_query(filepath, line)
        rows = self._safe_query(ql, "sanitizers")
        if not rows:
            return "=== SANITIZERS & GUARDS ===\n(none found in enclosing function)"

        lines: list[str] = []
        seen = set()
        for row in rows[:10]:
            key = (row.get("name", ""), row.get("line_number", ""))
            if key in seen:
                continue
            seen.add(key)
            kind = row.get("kind", "sanitizer").title()
            name = row.get("name", "?")
            ln = row.get("line_number", "?")
            code = row.get("code", "")
            lines.append(f"{kind}: {name} @ line {ln}" +
                         (f"  ({code})" if code else ""))

        return "=== SANITIZERS & GUARDS ===\n" + "\n".join(lines)

    # -- 3. scope ------------------------------------------------------

    def _run_scope(self, filepath: str, line: int) -> str | None:
        ql = scope_query(filepath, line)
        rows = self._safe_query(ql, "scope")
        if not rows:
            return None

        row = rows[0]
        kind = row.get("kind", "function")
        name = row.get("name", "?")
        start = row.get("start_line", "?")
        end = row.get("end_line", "?")
        params = row.get("parameters", "")
        decorators = row.get("decorators", "")

        parts = [f"{kind.title()}: {name}({params}) @ lines {start}-{end}"]
        if decorators:
            parts.append(f"Decorators: {decorators}")

        return "=== ENCLOSING SCOPE ===\n" + "\n".join(parts)

    # -- 4. call graph -------------------------------------------------

    def _run_call_graph(self, filepath: str, line: int) -> str | None:
        ql = call_graph_query(filepath, line)
        rows = self._safe_query(ql, "call_graph")
        if not rows:
            return None

        callers: list[str] = []
        callees: list[str] = []
        for row in rows:
            entry = f"{row.get('name', '?')} @ {row.get('file', '?')}:{row.get('line_number', '?')}"
            if row.get("direction") == "caller":
                callers.append(entry)
            else:
                callees.append(entry)

        lines = ["Callers: " + (", ".join(callers[:10])
                                if callers else "(none)")]
        lines.append(
            "Callees: " + (", ".join(callees[:10]) if callees else "(none)"))

        return "=== CALL GRAPH ===\n" + "\n".join(lines)

    # -- 5. sink classification ----------------------------------------

    def _run_sink_classification(self, filepath: str, line: int) -> str | None:
        ql = sink_classification_query(filepath, line)
        rows = self._safe_query(ql, "sink_classification")
        if not rows:
            return None

        seen = set()
        lines: list[str] = []
        for row in rows[:5]:
            kind = row.get("sink_kind", "?")
            if kind in seen:
                continue
            seen.add(kind)
            api = row.get("api_name", "?")
            lines.append(f"Sink type: {kind}")
            lines.append(f"API: {api}")

        return "=== SINK CLASSIFICATION ===\n" + "\n".join(lines)

    # -- 6. source classification --------------------------------------

    def _run_source_classification(self, filepath: str, line: int) -> str | None:
        ql = source_classification_query(filepath, line)
        rows = self._safe_query(ql, "source_classification")
        if not rows:
            return None

        seen = set()
        lines: list[str] = []
        for row in rows[:5]:
            kind = row.get("source_kind", "?")
            label = row.get("label", "?")
            key = (kind, label)
            if key in seen:
                continue
            seen.add(key)
            fl = row.get("file", "?")
            ln = row.get("line_number", "?")
            lines.append(f"Source: {label} @ {fl}:{ln}  ({kind})")

        return "=== SOURCE CLASSIFICATION ===\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove temporary database files."""
        self.client.cleanup()
