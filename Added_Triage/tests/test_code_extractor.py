"""Tests for code extractor."""

from pathlib import Path

from vuln_triage.code.extractor import CodeExtractor
from vuln_triage.sarif.models import Finding, Location

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PROJECT = FIXTURES / "sample_project"


def _make_finding(file_path: str, start_line: int, end_line: int | None = None) -> Finding:
    return Finding(
        rule_id="TEST",
        message="test",
        severity="medium",
        level="warning",
        location=Location(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line or start_line,
        ),
    )


def test_extract_snippet():
    """Extract a code snippet with context."""
    extractor = CodeExtractor(SAMPLE_PROJECT, context_lines=5, max_lines=50)
    finding = _make_finding("app.py", 20)

    snippet = extractor.extract(finding)

    assert ">>>" in snippet  # finding line is marked
    assert "cursor.execute" in snippet
    assert "SELECT * FROM users" in snippet


def test_extract_function():
    """Extract the enclosing function."""
    extractor = CodeExtractor(SAMPLE_PROJECT, context_lines=5)
    finding = _make_finding("app.py", 20)

    func = extractor.extract_function(finding)

    assert func is not None
    assert "def get_user" in func
    assert ">>>" in func


def test_extract_missing_file():
    """Handle missing source files gracefully."""
    extractor = CodeExtractor(SAMPLE_PROJECT, context_lines=5)
    finding = _make_finding("nonexistent.py", 1)

    snippet = extractor.extract(finding)

    assert "not found" in snippet.lower()


def test_extract_marks_correct_line():
    """Ensure the correct line is marked with >>>."""
    extractor = CodeExtractor(SAMPLE_PROJECT, context_lines=3, max_lines=50)
    finding = _make_finding("app.py", 42)

    snippet = extractor.extract(finding)

    for line in snippet.splitlines():
        if ">>>" in line:
            assert "42" in line.split(">>>")[0]
