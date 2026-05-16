"""Tests for SARIF parser."""

from pathlib import Path

from vuln_triage.sarif.parser import parse_sarif

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_sample_sarif():
    """Parse the sample SARIF file and verify findings."""
    findings = parse_sarif(FIXTURES / "sample.sarif.json")

    assert len(findings) == 4

    # First finding: SQL injection (true positive)
    f0 = findings[0]
    assert f0.rule_id == "B608"
    assert f0.file_path == "app.py"
    assert f0.start_line == 20
    assert f0.severity == "medium"
    assert f0.tool_name == "bandit"
    assert "SQL injection" in f0.message

    # Second finding: SQL injection (false positive — parameterized)
    f1 = findings[1]
    assert f1.rule_id == "B608"
    assert f1.start_line == 30

    # Third: eval (true positive)
    f2 = findings[2]
    assert f2.rule_id == "B307"
    assert f2.start_line == 42
    assert f2.severity == "high"  # error level

    # Fourth: eval with validation (false positive)
    f3 = findings[3]
    assert f3.rule_id == "B307"
    assert f3.start_line == 51


def test_parse_empty_sarif(tmp_path):
    """Handle SARIF with no results gracefully."""
    sarif = tmp_path / "empty.sarif.json"
    sarif.write_text('{"version": "2.1.0", "runs": [{"tool": {"driver": {"name": "test"}}, "results": []}]}')

    findings = parse_sarif(sarif)
    assert findings == []


def test_parse_sarif_no_locations(tmp_path):
    """Skip results that have no locations."""
    sarif = tmp_path / "noloc.sarif.json"
    sarif.write_text('''
    {
      "version": "2.1.0",
      "runs": [{
        "tool": {"driver": {"name": "test", "rules": []}},
        "results": [
          {"ruleId": "R001", "level": "warning", "message": {"text": "test"}, "locations": []},
          {"ruleId": "R002", "level": "error", "message": {"text": "found it"},
           "locations": [{"physicalLocation": {"artifactLocation": {"uri": "foo.py"}, "region": {"startLine": 10}}}]}
        ]
      }]
    }
    ''')

    findings = parse_sarif(sarif)
    assert len(findings) == 1
    assert findings[0].rule_id == "R002"
