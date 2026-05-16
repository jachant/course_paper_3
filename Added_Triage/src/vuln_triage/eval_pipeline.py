"""Evaluation pipeline — runs the triage model on a labelled dataset
and compares predictions against ground-truth labels.

Dataset format (concatenated JSON objects, one per entry):
    {"idx": 0, "func": "```python\n...\n```", "target": 0, "project": "python"}

target mapping:
    0 → false_positive (safe / not exploitable)
    1 → true_positive  (real vulnerability)
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .config import AppConfig
from .codeql.codeql_analyzer import CodeQLAnalyzer
from .codeql.codeql_client import CodeQLError
from .llm.client import LLMClient, LLMError
from .sarif.models import Finding, Location, TriageVerdict

logger = logging.getLogger(__name__)

# ── Target ↔ verdict mapping ──────────────────────────────────────────

TARGET_TO_VERDICT: dict[int, str] = {
    0: "false_positive",
    1: "true_positive",
}

VERDICT_TO_TARGET: dict[str, int | None] = {
    "false_positive": 0,
    "true_positive": 1,
    "uncertain": None,  # no direct mapping
}


# ── Data structures ───────────────────────────────────────────────────


@dataclass
class DatasetEntry:
    """One record from the labelled dataset."""

    idx: int
    func: str  # raw code (backticks stripped)
    target: int  # 0 = FP, 1 = TP
    project: str
    filename: str = ""  # populated after writing to temp dir


@dataclass
class EvalResult:
    """Result of evaluating a single dataset entry."""

    idx: int
    original_target: int
    original_label: str  # "true_positive" / "false_positive"
    predicted_verdict: str  # model's verdict
    confidence: float
    reasoning: str
    correct: bool  # predicted matches original
    tokens_used: int = 0
    error: str | None = None


@dataclass
class EvalReport:
    """Aggregated evaluation report."""

    dataset_path: str
    llm_model: str
    total: int = 0
    correct: int = 0
    incorrect: int = 0
    uncertain_count: int = 0
    errors: int = 0
    results: list[EvalResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Accuracy excluding uncertain predictions."""
        decided = self.correct + self.incorrect
        return self.correct / decided if decided > 0 else 0.0

    @property
    def accuracy_with_uncertain(self) -> float:
        """Accuracy counting uncertain as incorrect."""
        return self.correct / self.total if self.total > 0 else 0.0


# ── Dataset parsing ───────────────────────────────────────────────────


def _strip_code_fences(func_text: str) -> str:
    """Remove ```python ... ``` wrappers from code snippets."""
    text = func_text.strip()
    # Remove opening fence
    text = re.sub(r"^```\w*\n?", "", text)
    # Remove closing fence
    text = re.sub(r"\n?```\s*$", "", text)
    return text


def parse_dataset(path: Path) -> list[DatasetEntry]:
    """Parse the dataset file (concatenated JSON objects).

    The file contains multiple JSON objects (not a JSON array, not JSONL).
    Each object's ``func`` field may contain multi-line code with special
    characters that make streaming JSON decoding unreliable.  We locate
    object boundaries via ``{"idx": N`` regex and parse each chunk
    individually.
    """
    content = path.read_text(encoding="utf-8")

    # Find the start position of every top-level object
    pattern = re.compile(r'\{\s*"idx"\s*:\s*\d+')
    starts = [m.start() for m in pattern.finditer(content)]

    entries: list[DatasetEntry] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(content)
        chunk = content[start:end].rstrip().rstrip(",").rstrip()
        # Ensure we capture up to the last closing brace
        last_brace = chunk.rfind("}")
        if last_brace >= 0:
            chunk = chunk[: last_brace + 1]
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse object at pos %d: %s", start, exc)
            continue

        entries.append(
            DatasetEntry(
                idx=obj["idx"],
                func=_strip_code_fences(obj["func"]),
                target=obj["target"],
                project=obj.get("project", "python"),
            )
        )

    logger.info("Parsed %d entries from %s", len(entries), path)
    return entries


# ── Temp source tree ──────────────────────────────────────────────────


def write_snippets_to_dir(
    entries: list[DatasetEntry], base_dir: Path
) -> None:
    """Write each code snippet to its own .py file inside *base_dir*."""
    base_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        fname = f"snippet_{entry.idx:05d}.py"
        filepath = base_dir / fname
        filepath.write_text(entry.func, encoding="utf-8")
        entry.filename = fname


def _make_finding(entry: DatasetEntry, source_dir: Path) -> Finding:
    """Create a synthetic Finding for the dataset entry."""
    code_lines = entry.func.splitlines()
    return Finding(
        rule_id=f"DATASET-{entry.idx}",
        message="Dataset entry for evaluation",
        severity="medium",
        level="warning",
        location=Location(
            file_path=str(source_dir / entry.filename),
            start_line=1,
            end_line=len(code_lines),
        ),
        tool_name="dataset",
        tool_version="1.0",
    )


# ── Main evaluation pipeline ─────────────────────────────────────────


class EvalPipeline:
    """Runs the triage model on a labelled dataset and measures accuracy."""

    _codeql_lock = threading.Lock()

    def __init__(self, config: AppConfig):
        self.config = config

    @staticmethod
    def _run_codeql_analysis(
        codeql_analyzer: CodeQLAnalyzer, finding: Finding
    ) -> str | None:
        with EvalPipeline._codeql_lock:
            return codeql_analyzer.analyze(finding)

    def run(
        self,
        dataset_path: Path,
        output_path: Path,
        *,
        skip_graph: bool = False,
        limit: int | None = None,
    ) -> EvalReport:
        """Synchronous entry point."""
        return asyncio.run(
            self.run_async(
                dataset_path,
                output_path,
                skip_graph=skip_graph,
                limit=limit,
            )
        )

    async def run_async(
        self,
        dataset_path: Path,
        output_path: Path,
        *,
        skip_graph: bool = False,
        limit: int | None = None,
    ) -> EvalReport:
        """Run the full evaluation pipeline."""

        # 1. Parse dataset and shuffle for balanced sampling
        entries = parse_dataset(dataset_path)
        random.seed(42)
        random.shuffle(entries)
        if limit and limit < len(entries):
            entries = entries[:limit]
            logger.info("Limited to %d entries (shuffled)", limit)

        # 2. Write snippets to a temp directory
        tmp_dir = Path(tempfile.mkdtemp(prefix="eval_src_"))
        try:
            write_snippets_to_dir(entries, tmp_dir)
            logger.info("Wrote %d snippets to %s", len(entries), tmp_dir)

            # 3. Initialize CodeQL (single DB for all snippets)
            codeql_analyzer: CodeQLAnalyzer | None = None
            if not skip_graph:
                try:
                    codeql_analyzer = CodeQLAnalyzer(
                        tmp_dir, self.config.codeql
                    )
                    logger.info("CodeQL analyzer initialized for evaluation")
                except CodeQLError as e:
                    logger.warning(
                        "CodeQL init failed, proceeding without graph: %s", e
                    )

            # 4. Initialize LLM
            llm_client = LLMClient(self.config.llm)

            # 5. Process entries
            semaphore = asyncio.Semaphore(
                self.config.pipeline.max_concurrent_findings
            )
            results: list[EvalResult] = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
            ) as progress:
                task = progress.add_task(
                    "Evaluating dataset…", total=len(entries)
                )

                async def process_one(entry: DatasetEntry) -> EvalResult:
                    async with semaphore:
                        result = await self._process_entry(
                            entry, tmp_dir, codeql_analyzer, llm_client
                        )
                        progress.advance(task)
                        return result

                results = await asyncio.gather(
                    *[process_one(e) for e in entries]
                )

            # 6. Build report
            report = self._build_report(
                dataset_path, results
            )

            # 7. Save
            self._save_report(report, output_path)

            # Cleanup CodeQL
            if codeql_analyzer:
                codeql_analyzer.cleanup()

            return report

        finally:
            # Cleanup temp source directory
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _process_entry(
        self,
        entry: DatasetEntry,
        source_dir: Path,
        codeql_analyzer: CodeQLAnalyzer | None,
        llm_client: LLMClient,
    ) -> EvalResult:
        """Process a single dataset entry."""
        log_prefix = f"[idx={entry.idx}]"
        finding = _make_finding(entry, source_dir)

        # Code snippet is the entry itself
        code_snippet = entry.func

        # CodeQL analysis
        codeql_context: str | None = None
        if codeql_analyzer:
            try:
                codeql_context = await asyncio.to_thread(
                    self._run_codeql_analysis, codeql_analyzer, finding
                )
                if codeql_context:
                    logger.debug("%s CodeQL context extracted", log_prefix)
            except Exception as e:
                logger.warning("%s CodeQL error: %s", log_prefix, e)

        # LLM classification (eval-specific prompt, no SARIF metadata)
        original_label = TARGET_TO_VERDICT[entry.target]
        try:
            verdict, tokens = await llm_client.classify_code(
                code_snippet, codeql_context
            )

            predicted = verdict.verdict
            # For correctness: compare predicted verdict to original label
            # "uncertain" counts as incorrect
            correct = predicted == original_label

            logger.info(
                "%s predicted=%s actual=%s correct=%s (conf=%.2f)",
                log_prefix,
                predicted,
                original_label,
                correct,
                verdict.confidence,
            )

            return EvalResult(
                idx=entry.idx,
                original_target=entry.target,
                original_label=original_label,
                predicted_verdict=predicted,
                confidence=verdict.confidence,
                reasoning=verdict.reasoning,
                correct=correct,
                tokens_used=tokens,
            )

        except LLMError as e:
            logger.error("%s LLM error: %s", log_prefix, e)
            return EvalResult(
                idx=entry.idx,
                original_target=entry.target,
                original_label=original_label,
                predicted_verdict="error",
                confidence=0.0,
                reasoning=f"LLM call failed: {e}",
                correct=False,
                error=str(e),
            )

    def _build_report(
        self,
        dataset_path: Path,
        results: list[EvalResult],
    ) -> EvalReport:
        """Aggregate individual results into an EvalReport."""
        report = EvalReport(
            dataset_path=str(dataset_path),
            llm_model=self.config.llm.model,
            total=len(results),
            results=results,
        )

        for r in results:
            if r.error:
                report.errors += 1
            elif r.predicted_verdict == "uncertain":
                report.uncertain_count += 1
            elif r.correct:
                report.correct += 1
            else:
                report.incorrect += 1

        return report

    def _save_report(self, report: EvalReport, output_path: Path) -> None:
        """Save the evaluation report as JSON."""
        data = {
            "evaluation": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset": report.dataset_path,
                "llm_model": report.llm_model,
                "total_entries": report.total,
                "correct": report.correct,
                "incorrect": report.incorrect,
                "uncertain": report.uncertain_count,
                "errors": report.errors,
                "accuracy": round(report.accuracy * 100, 2),
                "accuracy_with_uncertain_as_wrong": round(
                    report.accuracy_with_uncertain * 100, 2
                ),
            },
            "results": [
                {
                    "idx": r.idx,
                    "original_target": r.original_target,
                    "original_label": r.original_label,
                    "predicted_verdict": r.predicted_verdict,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "correct": r.correct,
                    "tokens_used": r.tokens_used,
                    **({"error": r.error} if r.error else {}),
                }
                for r in report.results
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Evaluation report saved to %s", output_path)
