"""JSON triage report generator."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from vuln_triage.sarif.models import TriageReport, TriageResult

logger = logging.getLogger(__name__)


def generate_report(report: TriageReport) -> dict:
    """Convert a TriageReport into a JSON-serializable dict."""
    return {
        "report": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sarif_file": report.sarif_file,
            "source_path": report.source_path,
            "llm_model": report.llm_model,
            "total_findings": report.total,
            "true_positives": report.true_positives,
            "false_positives": report.false_positives,
            "uncertain": report.uncertain,
            "errors": report.errors,
        },
        "findings": [_serialize_result(r) for r in report.results],
    }


def save_report(report: TriageReport, output_path: Path) -> None:
    """Generate and save the triage report as JSON."""
    data = generate_report(report)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", output_path)


def _serialize_result(result: TriageResult) -> dict:
    """Serialize a single TriageResult."""
    entry = {
        "rule_id": result.finding.rule_id,
        "file": result.finding.file_path,
        "line": result.finding.start_line,
        "severity": result.finding.severity,
        "tool_message": result.finding.message,
        "verdict": result.verdict.verdict,
        "confidence": result.verdict.confidence,
        "reasoning": result.verdict.reasoning,
        "llm_model": result.llm_model,
        "tokens_used": result.llm_tokens_used,
    }
    if result.error:
        entry["error"] = result.error
    return entry
