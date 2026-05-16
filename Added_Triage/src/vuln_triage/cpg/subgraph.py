"""CPG subgraph extraction and serialization for LLM prompts."""
from __future__ import annotations
import logging
from pathlib import Path
from vuln_triage.config import JoernConfig
from vuln_triage.sarif.models import Finding
from .joern_client import JoernClient, JoernError
from .queries import (
    CALL_CONTEXT,
    DATAFLOW_AT_LINE,
    METHOD_AT_LINE,
    METHOD_CALLS,
    build_query,
)

logger = logging.getLogger(__name__)


class CPGAnalyzer:
    """Extracts CPG subgraphs relevant to vulnerability findings."""

    def __init__(self, source_root: Path, config: JoernConfig):
        self.source_root = source_root
        self.client = JoernClient(config)
        self._cpg_ready = False

    def ensure_cpg(self) -> None:
        """Generate CPG for the source root if not already done."""
        if not self._cpg_ready:
            self.client.generate_cpg(self.source_root)
            self._cpg_ready = True

    def analyze(self, finding: Finding) -> str | None:
        """Extract CPG context for a finding.

        Returns a text representation of the relevant subgraph,
        or None if CPG analysis fails.
        """
        try:
            self.ensure_cpg()
        except JoernError as e:
            logger.error("CPG generation failed: %s", e)
            return None

        filename = finding.file_path
        line = finding.start_line

        # Run all queries in batch for efficiency (single JVM invocation)
        queries = [
            build_query(METHOD_AT_LINE, filename, line),
            build_query(DATAFLOW_AT_LINE, filename, line),
            build_query(CALL_CONTEXT, filename, line),
            build_query(METHOD_CALLS, filename, line),
        ]

        try:
            results = self.client.run_batch_queries(queries)
        except JoernError as e:
            logger.warning("CPG query failed for %s:%d — %s",
                           filename, line, e)
            return None

        # Pad results if some queries didn't produce output
        while len(results) < 4:
            results.append("")

        method_info, dataflow_info, call_context, method_calls = results

        return self._format_context(
            method_info=method_info,
            dataflow_info=dataflow_info,
            call_context=call_context,
            method_calls=method_calls,
            filename=filename,
            line=line,
        )

    # Strings that indicate an empty / failed query result
    _EMPTY_MARKERS = {
        "", "List()", "None", "()", "Vector()",
        "NO_SINKS_FOUND_AT_LINE", "NO_DATAFLOW_PATHS_FOUND",
        "DATAFLOW_INIT_FAILED", "NO_ENCLOSING_METHOD_FOUND",
    }

    def _is_useful(self, text: str) -> bool:
        """Return True if the query result contains meaningful data."""
        stripped = text.strip()
        if stripped in self._EMPTY_MARKERS:
            return False
        if stripped.startswith("DATAFLOW_ERROR:"):
            logger.debug("Data flow error: %s", stripped)
            return False
        if stripped.startswith("DATAFLOW_INIT_FAILED"):
            logger.debug("Data flow init failed: %s", stripped)
            return False
        return bool(stripped)

    def _validate_method_info(self, method_info: str, filename: str, line: int) -> str:
        """Validate that the method info actually corresponds to the finding location."""
        if not method_info or "METHOD: " not in method_info:
            return method_info

        # Parse "METHOD: name | lines START-END"
        try:
            parts = method_info.split(" | lines ")
            if len(parts) >= 2:
                line_range = parts[1].split("\n")[0].strip()
                start_str, end_str = line_range.split("-")
                method_start = int(start_str)
                method_end = int(end_str)

                # Check if the target line is within the method's range
                if not (method_start <= line <= method_end):
                    logger.warning(
                        "CPG method mismatch: %s reports lines %d-%d "
                        "but finding is at %s:%d",
                        parts[0].strip(), method_start, method_end,
                        filename, line,
                    )
                    # Return with a warning annotation for the LLM
                    return (
                        f"⚠️ NOTE: Method range ({method_start}-{method_end}) "
                        f"does not contain target line {line}. "
                        f"CPG method resolution may be inaccurate.\n"
                        f"{method_info}"
                    )
        except (ValueError, IndexError):
            pass  # Can't parse — return as-is

        return method_info

    def _format_context(
        self,
        method_info: str,
        dataflow_info: str,
        call_context: str,
        method_calls: str,
        filename: str,
        line: int,
    ) -> str:
        """Format CPG query results into a readable text block for the LLM."""
        sections = []

        if self._is_useful(method_info):
            validated_info = self._validate_method_info(
                method_info, filename, line)
            sections.append(f"=== ENCLOSING METHOD ===\n{validated_info}")

        if self._is_useful(dataflow_info):
            sections.append(f"=== DATA FLOW PATHS ===\n{dataflow_info}")

        if self._is_useful(call_context) and "CALLERS:" in call_context:
            sections.append(f"=== CALL CONTEXT ===\n{call_context}")

        if self._is_useful(method_calls):
            sections.append(f"=== METHOD CALLS ===\n{method_calls}")

        if not sections:
            return None

        header = f"CPG Analysis for {filename}:{line}"
        return f"{header}\n{'=' * len(header)}\n\n" + "\n\n".join(sections)

    def cleanup(self):
        """Clean up temporary files."""
        self.client.cleanup()
