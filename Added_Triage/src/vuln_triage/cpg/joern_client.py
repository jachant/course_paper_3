"""Joern CLI client — generates CPGs and runs CPGQL queries via subprocess."""
from __future__ import annotations
import json
import logging
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from vuln_triage.config import JoernConfig

logger = logging.getLogger(__name__)


class JoernError(Exception):
    """Raised when Joern command fails."""
    pass


class JoernClient:
    """Interfaces with Joern via CLI (joern-parse, joern)."""

    # Thread-safe counter for script names
    _script_counter_lock = threading.Lock()
    _script_counter = 0

    def __init__(self, config: JoernConfig):
        self.config = config
        self._joern_parse = self._find_binary("joern-parse")
        self._joern = self._find_binary("joern")
        self._cpg_dir = Path(tempfile.mkdtemp(prefix="vuln_triage_cpg_"))
        self._cpg_built_for: str | None = None

    def _find_binary(self, name: str) -> str:
        """Locate a Joern binary on PATH or in configured joern_path."""
        if self.config.joern_path:
            candidate = Path(self.config.joern_path) / name
            if candidate.exists():
                return str(candidate)

        found = shutil.which(name)
        if found:
            return found

        raise JoernError(
            f"'{name}' not found. Install Joern or set joern.joern_path in config.  "
            f"See: https://docs.joern.io/installation "
        )

    @property
    def cpg_path(self) -> Path:
        return self._cpg_dir / "cpg.bin"

    def generate_cpg(self, source_path: Path) -> Path:
        """Generate a CPG for the given source directory using joern-parse.

        Returns path to the generated cpg.bin.
        """
        source_str = str(source_path.resolve())

        # Skip if already built for this path
        if self._cpg_built_for == source_str and self.cpg_path.exists():
            logger.info("Reusing existing CPG for %s", source_path)
            return self.cpg_path

        logger.info("Generating CPG for %s ...", source_path)
        cmd = [
            self._joern_parse,
            source_str,
            "--language", "pythonsrc",  # Fixed: removed trailing spaces
            "--output", str(self.cpg_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.parse_timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise JoernError(
                f"CPG generation timed out after {self.config.parse_timeout}s"
            ) from e

        if result.returncode != 0:
            raise JoernError(
                f"joern-parse failed (exit {result.returncode}):\n{result.stderr}"
            )

        self._cpg_built_for = source_str
        logger.info("CPG generated: %s", self.cpg_path)
        return self.cpg_path

    def run_batch_queries(self, queries: list[str]) -> list[str]:
        if not self.cpg_path.exists():
            raise JoernError(
                "No CPG generated yet. Call generate_cpg() first."
            )

        delimiter = "===QUERY_RESULT_DELIMITER==="
        parts = []
        for q in queries:
            parts.append(f'println("{delimiter}")')
            parts.append(f'println({q})')

        combined = "\n".join(parts)
        script = self._create_script(combined)

        cmd = [
            self._joern,
            "--script", str(script),
            "--nocolors",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.query_timeout * len(queries),
                env=self._get_joern_env(),
            )
        except subprocess.TimeoutExpired as e:
            raise JoernError("Batch query timed out") from e
        finally:
            script.unlink(missing_ok=True)

        # Check for data flow initialization failure in stderr
        if "DATAFLOW_INIT_FAILED" in (result.stderr or ""):
            logger.warning("Joern data flow overlay failed to initialize")

        if result.returncode != 0:
            logger.debug("=== JOERN STDOUT ===\n%s", result.stdout[:2000])
            logger.debug("=== JOERN STDERR ===\n%s", result.stderr[:2000])
            raise JoernError(
                f"Batch query failed (exit {result.returncode}):\n"
                f"STDERR: {result.stderr[:500]}\n"
                f"STDOUT: {result.stdout[:500]}"
            )

        raw_results = result.stdout.split(delimiter)
        results = [self._clean_output(r.strip()) for r in raw_results[1:]]

        # Pad results if some queries didn't produce output
        while len(results) < len(queries):
            results.append("")

        return results

    @staticmethod
    def _get_joern_env() -> dict:
        """Build environment for Joern subprocess with reduced logging."""
        import os
        env = os.environ.copy()
        # Suppress Joern's verbose JVM logging
        env.setdefault("JAVA_OPTS", "")
        if "-Dlog.level" not in env["JAVA_OPTS"]:
            env["JAVA_OPTS"] += " -Dlog.level=ERROR"
        return env

    # Patterns for Joern internal messages that leak into stdout.
    _NOISE_RE = re.compile(
        r"^\[INFO\s*\].*$|"
        r"^\[INFO\s*\n\s*\].*$|"
        r"^\[WARNING\s*\].*$|"
        r"^\[WARNING\s*\n\s*\].*$|"
        r"^SLF4J:.*$|"
        r"^replpp\..*$|"
        r"^writing to storage.*$|"
        r"^closed graph.*$|"
        r"^For input string.*$|"
        r"^log4j:.*$|"
        r"^WARNING:.*$|"
        r"^Compiling.*$",
        re.MULTILINE,
    )

    _MULTILINE_LOG_RE = re.compile(
        r"\[INFO\s*\n\s*\][^\n]*",
        re.MULTILINE,
    )

    @classmethod
    def _clean_output(cls, text: str) -> str:
        """Remove Joern internal log lines from query output."""
        # First handle multi-line log entries like "[INFO\n        ] msg"
        cleaned = cls._MULTILINE_LOG_RE.sub("", text)
        # Then handle single-line patterns
        cleaned = cls._NOISE_RE.sub("", cleaned)
        # Collapse multiple blank lines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _create_script(self, query: str) -> Path:
        # Escape backslashes in CPG path for Scala string literal
        cpg_path_escaped = str(self.cpg_path).replace("\\", "\\\\")

        script_content = (
            'import io.shiftleft.semanticcpg.language._\n'
            'import io.joern.dataflowengineoss.language._\n'
            'import io.joern.dataflowengineoss.queryengine.EngineContext\n'
            '\n'
            'System.setProperty("log.level", "ERROR")\n'
            '\n'
            '// Load the pre-built CPG\n'
            f'importCpg("{cpg_path_escaped}")\n'
            '\n'
            '// Initialize data flow overlay; print marker on failure\n'
            'val dataflowReady = try {\n'
            '  run.ossdataflow\n'
            '  true\n'
            '} catch {\n'
            '  case e: Throwable =>\n'
            '    System.err.println("DATAFLOW_INIT_FAILED: " + e.getMessage)\n'
            '    false\n'
            '}\n'
            '\n'
            f'{query}\n'
            '\n'
            '// Clean up: close the CPG to avoid workspace pollution\n'
            'workspace.reset\n'
        )

        # Thread-safe unique script name
        with JoernClient._script_counter_lock:
            JoernClient._script_counter += 1
            counter = JoernClient._script_counter

        script_path = self._cpg_dir / f"query_script_{counter}.sc"
        script_path.write_text(script_content, encoding="utf-8")
        return script_path

    def cleanup(self):
        """Remove temporary CPG files and Joern workspace artifacts."""
        if self._cpg_dir.exists():
            shutil.rmtree(self._cpg_dir, ignore_errors=True)

        # Clean up Joern workspace directories that accumulate
        # from importCpg calls (cpg.bin1/, cpg.bin2/, etc.)
        workspace_dir = Path("workspace")
        if workspace_dir.exists():
            for entry in workspace_dir.iterdir():
                if entry.is_dir() and entry.name.startswith("cpg.bin"):
                    shutil.rmtree(entry, ignore_errors=True)
                    logger.debug("Cleaned up Joern workspace: %s", entry)

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass  # Suppress errors during garbage collection
