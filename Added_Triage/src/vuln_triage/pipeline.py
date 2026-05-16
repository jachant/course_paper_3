"""Main triage pipeline — orchestrates all processing steps."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .code.extractor import CodeExtractor
from .config import AppConfig
from .codeql.codeql_analyzer import CodeQLAnalyzer
from .codeql.codeql_client import CodeQLError
from .llm.client import LLMClient, LLMError
from .sarif.models import Finding, TriageReport, TriageResult, TriageVerdict
from .sarif.parser import parse_sarif

logger = logging.getLogger(__name__)


class TriagePipeline:
    """Orchestrates the full vulnerability triage pipeline."""

    # Lock to serialize CodeQL queries — codeql CLI can have issues
    # with concurrent subprocess calls on the same database.
    _codeql_lock = threading.Lock()

    def __init__(self, config: AppConfig):
        self.config = config

    @staticmethod
    def _run_codeql_analysis(
        codeql_analyzer: CodeQLAnalyzer, finding: Finding
    ) -> str | None:
        """Run CodeQL analysis under a lock to prevent concurrent access."""
        with TriagePipeline._codeql_lock:
            return codeql_analyzer.analyze(finding)

    def run(
        self,
        sarif_path: Path,
        source_path: Path,
        output_path: Path | None = None,
        skip_graph: bool = False,
    ) -> TriageReport:
        """Run the full triage pipeline synchronously.

        Args:
            sarif_path: Path to the SARIF report file.
            source_path: Path to the source code root directory.
            output_path: Optional path to save the JSON report.
            skip_graph: If True, skip CodeQL graph analysis (code-only mode).

        Returns:
            TriageReport with results for all findings.
        """
        return asyncio.run(
            self.run_async(sarif_path, source_path, output_path, skip_graph)
        )

    async def run_async(
        self,
        sarif_path: Path,
        source_path: Path,
        output_path: Path | None = None,
        skip_graph: bool = False,
    ) -> TriageReport:
        """Run the full triage pipeline asynchronously."""
        # Step 1: Parse SARIF
        logger.info("Parsing SARIF report: %s", sarif_path)
        findings = parse_sarif(sarif_path)
        if not findings:
            logger.warning("No findings in SARIF report")
            return TriageReport(
                sarif_file=str(sarif_path),
                source_path=str(source_path),
                llm_model=self.config.llm.model,
            )

        logger.info("Found %d findings to triage", len(findings))

        # Step 2: Initialize components
        extractor = CodeExtractor(
            source_root=source_path,
            context_lines=self.config.extraction.context_lines,
            max_lines=self.config.extraction.max_snippet_lines,
        )

        codeql_analyzer: CodeQLAnalyzer | None = None
        if not skip_graph:
            try:
                codeql_analyzer = CodeQLAnalyzer(source_path, self.config.codeql)
                logger.info("CodeQL analyzer initialized")
            except CodeQLError as e:
                logger.warning(
                    "CodeQL initialization failed, proceeding without graph analysis: %s", e
                )
                codeql_analyzer = None

        llm_client = LLMClient(self.config.llm)

        # Step 3: Process each finding
        semaphore = asyncio.Semaphore(
            self.config.pipeline.max_concurrent_findings)
        results: list[TriageResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(
                "Triaging findings...", total=len(findings))

            async def process_one(finding: Finding) -> TriageResult:
                async with semaphore:
                    result = await self._process_finding(
                        finding, extractor, codeql_analyzer, llm_client
                    )
                    progress.advance(task)
                    return result

            results = await asyncio.gather(
                *[process_one(f) for f in findings]
            )

        # Step 4: Build report
        report = TriageReport(
            sarif_file=str(sarif_path),
            source_path=str(source_path),
            llm_model=self.config.llm.model,
            results=list(results),
        )

        # Step 5: Save if output path given
        if output_path:
            from .report.generator import save_report
            save_report(report, output_path)

        # Cleanup
        if codeql_analyzer:
            codeql_analyzer.cleanup()

        return report

    async def _process_finding(
        self,
        finding: Finding,
        extractor: CodeExtractor,
        codeql_analyzer: CodeQLAnalyzer | None,
        llm_client: LLMClient,
    ) -> TriageResult:
        """Process a single finding through the pipeline."""
        log_prefix = f"[{finding.rule_id} @ {finding.file_path}:{finding.start_line}]"

        # Extract code
        code_snippet = extractor.extract(finding)
        func_snippet = extractor.extract_function(finding)
        if func_snippet and f"{finding.start_line:4d}" in func_snippet:
            code_snippet = func_snippet

        # CodeQL analysis (run in thread to avoid blocking event loop)
        codeql_context: str | None = None
        if codeql_analyzer:
            try:
                codeql_context = await asyncio.to_thread(
                    self._run_codeql_analysis, codeql_analyzer, finding
                )
                if codeql_context:
                    logger.debug("%s CodeQL context extracted", log_prefix)
                else:
                    logger.debug("%s No CodeQL context available", log_prefix)
            except Exception as e:
                logger.warning("%s CodeQL analysis error: %s", log_prefix, e)

        # LLM triage
        try:
            verdict, tokens = await llm_client.triage_finding(
                finding, code_snippet, codeql_context
            )
            logger.info(
                "%s -> %s (confidence: %.2f)",
                log_prefix, verdict.verdict, verdict.confidence,
            )
            return TriageResult(
                finding=finding,
                verdict=verdict,
                code_snippet=code_snippet,
                codeql_context=codeql_context,
                llm_model=self.config.llm.model,
                llm_tokens_used=tokens,
            )
        except LLMError as e:
            logger.error("%s LLM error: %s", log_prefix, e)
            return TriageResult(
                finding=finding,
                verdict=TriageVerdict(
                    verdict="uncertain",
                    confidence=0.0,
                    reasoning=f"LLM call failed: {e}",
                ),
                code_snippet=code_snippet,
                codeql_context=codeql_context,
                llm_model=self.config.llm.model,
                error=str(e),
            )
