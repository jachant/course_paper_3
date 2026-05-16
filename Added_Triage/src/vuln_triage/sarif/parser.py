"""SARIF report parser — extracts findings from SARIF JSON."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import CodeFlowStep, Finding, Location

logger = logging.getLogger(__name__)


class SarifParseError(Exception):
    """Raised when SARIF report cannot be parsed."""


def parse_sarif(sarif_path: Path) -> list[Finding]:
    """Parse a SARIF file and return a list of Findings.

    Supports SARIF v2.1.0 schema.
    """
    try:
        with open(sarif_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise SarifParseError(f"Failed to read SARIF file: {e}") from e

    version = data.get("version", "")
    if not version.startswith("2.1"):
        logger.warning("SARIF version %s — expected 2.1.x, may not parse correctly", version)

    findings: list[Finding] = []

    for run in data.get("runs", []):
        tool_name, tool_version = _extract_tool_info(run)
        rules_map = _build_rules_map(run)

        for result in run.get("results", []):
            finding = _parse_result(result, rules_map, tool_name, tool_version)
            if finding:
                findings.append(finding)

    logger.info("Parsed %d findings from %s", len(findings), sarif_path.name)
    return findings


def _extract_tool_info(run: dict) -> tuple[str, str]:
    """Extract tool name and version from a SARIF run."""
    tool = run.get("tool", {})
    driver = tool.get("driver", {})
    return driver.get("name", "unknown"), driver.get("version", "")


def _build_rules_map(run: dict) -> dict[str, dict]:
    """Build a map of rule_id → rule metadata from SARIF run."""
    rules = {}
    driver = run.get("tool", {}).get("driver", {})
    for rule in driver.get("rules", []):
        rule_id = rule.get("id", "")
        if rule_id:
            rules[rule_id] = rule
    return rules


def _parse_result(
    result: dict,
    rules_map: dict[str, dict],
    tool_name: str,
    tool_version: str,
) -> Finding | None:
    """Parse a single SARIF result into a Finding."""
    rule_id = result.get("ruleId", "")
    if not rule_id:
        rule_id = result.get("rule", {}).get("id", "unknown")

    # Message
    message = result.get("message", {}).get("text", "")

    # Level / severity
    level = result.get("level", "warning")
    severity = _level_to_severity(level)

    # Enrich from rules metadata
    rule_meta = rules_map.get(rule_id, {})
    if not message:
        message = rule_meta.get("shortDescription", {}).get("text", "")

    # Primary location
    locations = result.get("locations", [])
    if not locations:
        logger.debug("Skipping result without locations: %s", rule_id)
        return None

    location = _parse_location(locations[0])
    if not location:
        logger.debug("Skipping result with unparseable location: %s", rule_id)
        return None

    # Code flows
    code_flows = _parse_code_flows(result.get("codeFlows", []))

    # Related locations
    related = []
    for rl in result.get("relatedLocations", []):
        loc = _parse_location(rl)
        if loc:
            related.append(loc)

    return Finding(
        rule_id=rule_id,
        message=message,
        severity=severity,
        level=level,
        location=location,
        code_flows=code_flows,
        related_locations=related,
        properties=result.get("properties", {}),
        tool_name=tool_name,
        tool_version=tool_version,
    )


def _parse_location(loc_wrapper: dict) -> Location | None:
    """Parse a SARIF location object into a Location."""
    phys = loc_wrapper.get("physicalLocation", loc_wrapper)
    artifact = phys.get("artifactLocation", {})
    region = phys.get("region", {})

    uri = artifact.get("uri", "")
    if not uri:
        return None

    # Strip file:// prefix if present
    if uri.startswith("file://"):
        uri = uri[7:]

    start_line = region.get("startLine", 1)
    end_line = region.get("endLine", start_line)
    start_col = region.get("startColumn")
    end_col = region.get("endColumn")

    return Location(
        file_path=uri,
        start_line=start_line,
        end_line=end_line,
        start_column=start_col,
        end_column=end_col,
    )


def _parse_code_flows(code_flows_data: list[dict]) -> list[list[CodeFlowStep]]:
    """Parse SARIF codeFlows into lists of CodeFlowSteps."""
    flows = []
    for cf in code_flows_data:
        for thread_flow in cf.get("threadFlows", []):
            steps = []
            for loc_entry in thread_flow.get("locations", []):
                location = _parse_location(loc_entry.get("location", {}))
                if location:
                    msg = loc_entry.get("location", {}).get("message", {}).get("text", "")
                    steps.append(CodeFlowStep(location=location, message=msg))
            if steps:
                flows.append(steps)
    return flows


def _level_to_severity(level: str) -> str:
    """Map SARIF level to a normalized severity."""
    mapping = {
        "error": "high",
        "warning": "medium",
        "note": "low",
        "none": "info",
    }
    return mapping.get(level, "medium")
