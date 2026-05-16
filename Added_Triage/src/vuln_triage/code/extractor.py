"""Source code snippet extraction based on SARIF locations."""

from __future__ import annotations

import logging
from pathlib import Path

from vuln_triage.sarif.models import Finding

logger = logging.getLogger(__name__)


class CodeExtractor:
    """Extracts source code snippets for findings."""

    def __init__(self, source_root: Path, context_lines: int = 30, max_lines: int = 100):
        self.source_root = source_root.resolve()
        self.context_lines = context_lines
        self.max_lines = max_lines

    def extract(self, finding: Finding) -> str:
        """Extract code snippet for a finding with surrounding context.

        Returns the snippet as a string with line numbers.
        """
        file_path = self._resolve_path(finding.file_path)
        if not file_path or not file_path.exists():
            logger.warning("Source file not found: %s", finding.file_path)
            return f"# Source file not found: {finding.file_path}"

        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            logger.error("Failed to read %s: %s", file_path, e)
            return f"# Error reading file: {e}"

        # Calculate range with context
        start = max(0, finding.start_line - 1 - self.context_lines)
        end = min(len(lines), finding.end_line + self.context_lines)

        # Cap total lines
        if end - start > self.max_lines:
            half = self.max_lines // 2
            center = (finding.start_line - 1 + finding.end_line) // 2
            start = max(0, center - half)
            end = min(len(lines), start + self.max_lines)

        snippet_lines = []
        for i in range(start, end):
            line_num = i + 1
            marker = " >>> " if finding.start_line <= line_num <= finding.end_line else "     "
            snippet_lines.append(f"{line_num:4d}{marker}{lines[i]}")

        return "\n".join(snippet_lines)

    def extract_function(self, finding: Finding) -> str | None:
        """Try to extract the full enclosing function for a finding.

        Uses simple heuristic: scan backwards for 'def ' or 'class '.
        Returns None if function boundary cannot be determined.
        """
        file_path = self._resolve_path(finding.file_path)
        if not file_path or not file_path.exists():
            return None

        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None

        target_line = finding.start_line - 1  # 0-indexed

        # Find function start: scan backwards for 'def '
        func_start = None
        func_indent = None
        for i in range(target_line, -1, -1):
            stripped = lines[i].lstrip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                func_start = i
                func_indent = len(lines[i]) - len(stripped)
                break

        if func_start is None:
            return None

        # Find function end: next line at same or lower indent (or EOF)
        func_end = len(lines)
        for i in range(func_start + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                continue  # skip blank lines
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= func_indent and line.strip():
                func_end = i
                break

        # Cap function length
        if func_end - func_start > self.max_lines:
            func_end = func_start + self.max_lines

        result_lines = []
        for i in range(func_start, func_end):
            line_num = i + 1
            marker = " >>> " if finding.start_line <= line_num <= finding.end_line else "     "
            result_lines.append(f"{line_num:4d}{marker}{lines[i]}")

        return "\n".join(result_lines)

    def _resolve_path(self, relative_path: str) -> Path | None:
        """Resolve a SARIF-relative path against the source root."""
        # Try as-is relative to source root
        candidate = self.source_root / relative_path
        if candidate.exists():
            return candidate.resolve()

        # Try stripping leading slashes
        stripped = relative_path.lstrip("/")
        candidate = self.source_root / stripped
        if candidate.exists():
            return candidate.resolve()

        # Try just the filename
        candidate = self.source_root / Path(relative_path).name
        if candidate.exists():
            return candidate.resolve()

        return None
