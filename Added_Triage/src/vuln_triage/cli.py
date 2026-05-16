"""CLI entrypoint for vuln-triage."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .eval_pipeline import EvalPipeline, EvalReport
from .pipeline import TriagePipeline
from .report.generator import generate_report, save_report

app = typer.Typer(
    name="vuln-triage",
    help="Automated vulnerability triage using CodeQL analysis and LLMs.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def triage(
    sarif: Path = typer.Argument(..., help="Path to SARIF report file", exists=True),
    source: Path = typer.Argument(..., help="Path to source code root directory", exists=True),
    output: Path = typer.Option(None, "--output", "-o", help="Output JSON report path"),
    config_file: Path = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    model: str = typer.Option(None, "--model", "-m", help="Override LLM model"),
    skip_graph: bool = typer.Option(False, "--skip-graph", help="Skip CodeQL graph analysis (code-only mode)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """Triage vulnerabilities from a SARIF report using CodeQL + LLM analysis."""
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_datefmt = "%H:%M:%S"

    # Console handler
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=log_datefmt,
    )

    # File handler — always writes DEBUG level to triage.log
    log_file = sarif.with_suffix(".log") if sarif else Path("triage.log")
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=log_datefmt))
    logging.getLogger().addHandler(file_handler)

    # Load config
    config = load_config(config_file)
    if model:
        config.llm.model = model

    # Default output path
    if not output:
        output = sarif.with_suffix(".triage.json")

    console.print(f"\n[bold]vuln-triage[/bold] v0.1.0\n")
    console.print(f"  SARIF:    {sarif}")
    console.print(f"  Source:   {source}")
    console.print(f"  Model:    {config.llm.model}")
    console.print(f"  Analysis: {'disabled' if skip_graph else 'CodeQL'}")
    console.print(f"  Output:   {output}\n")

    # Run pipeline
    pipeline = TriagePipeline(config)
    try:
        report = pipeline.run(sarif, source, output, skip_graph=skip_graph)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Print summary
    _print_summary(report)
    console.print(f"\n[green]Report saved to:[/green] {output}")


def _print_summary(report):
    """Print a summary table of triage results."""
    table = Table(title="Triage Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")

    table.add_row("Total findings", str(report.total))
    table.add_row("True positives", f"[red]{report.true_positives}[/red]")
    table.add_row("False positives", f"[green]{report.false_positives}[/green]")
    table.add_row("Uncertain", f"[yellow]{report.uncertain}[/yellow]")
    table.add_row("Errors", f"[red]{report.errors}[/red]" if report.errors else "0")

    console.print(table)

    # Detailed findings table
    if report.results:
        detail = Table(title="Findings Detail")
        detail.add_column("#", justify="right", width=4)
        detail.add_column("Rule", max_width=30)
        detail.add_column("File:Line", max_width=40)
        detail.add_column("Verdict", width=16)
        detail.add_column("Conf.", justify="right", width=6)

        for i, r in enumerate(report.results, 1):
            verdict_style = {
                "true_positive": "red",
                "false_positive": "green",
                "uncertain": "yellow",
            }.get(r.verdict.verdict, "white")

            detail.add_row(
                str(i),
                r.finding.rule_id,
                f"{r.finding.file_path}:{r.finding.start_line}",
                f"[{verdict_style}]{r.verdict.verdict}[/{verdict_style}]",
                f"{r.verdict.confidence:.0%}",
            )

        console.print(detail)


@app.command()
def evaluate(
    dataset: Path = typer.Argument(..., help="Path to labelled dataset JSON file", exists=True),
    output: Path = typer.Option(None, "--output", "-o", help="Output JSON report path"),
    config_file: Path = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    model: str = typer.Option(None, "--model", "-m", help="Override LLM model"),
    skip_graph: bool = typer.Option(False, "--skip-graph", help="Skip CodeQL graph analysis"),
    limit: int = typer.Option(None, "--limit", "-n", help="Process only first N entries"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """Evaluate the triage pipeline against a labelled dataset and measure accuracy."""
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_datefmt = "%H:%M:%S"

    logging.basicConfig(level=log_level, format=log_format, datefmt=log_datefmt)

    log_file = dataset.with_suffix(".eval.log")
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=log_datefmt))
    logging.getLogger().addHandler(file_handler)

    # Load config
    config = load_config(config_file)
    if model:
        config.llm.model = model

    # Default output path
    if not output:
        output = dataset.with_suffix(".eval_results.json")

    console.print(f"\n[bold]vuln-triage evaluate[/bold]\n")
    console.print(f"  Dataset:  {dataset}")
    console.print(f"  Model:    {config.llm.model}")
    console.print(f"  Analysis: {'disabled' if skip_graph else 'CodeQL'}")
    console.print(f"  Limit:    {limit or 'all'}")
    console.print(f"  Output:   {output}\n")

    # Run evaluation
    pipeline = EvalPipeline(config)
    try:
        report = pipeline.run(dataset, output, skip_graph=skip_graph, limit=limit)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Print summary
    _print_eval_summary(report)
    console.print(f"\n[green]Evaluation report saved to:[/green] {output}")


def _print_eval_summary(report: EvalReport):
    """Print evaluation results summary."""
    table = Table(title="Evaluation Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total entries", str(report.total))
    table.add_row("Correct", f"[green]{report.correct}[/green]")
    table.add_row("Incorrect", f"[red]{report.incorrect}[/red]")
    table.add_row("Uncertain", f"[yellow]{report.uncertain_count}[/yellow]")
    table.add_row("Errors", f"[red]{report.errors}[/red]" if report.errors else "0")
    table.add_row("", "")
    table.add_row(
        "Accuracy (excl. uncertain)",
        f"[bold cyan]{report.accuracy:.1%}[/bold cyan]",
    )
    table.add_row(
        "Accuracy (uncertain = wrong)",
        f"[cyan]{report.accuracy_with_uncertain:.1%}[/cyan]",
    )

    console.print(table)

    # Confusion breakdown
    if report.results:
        tp_correct = sum(
            1 for r in report.results
            if r.original_label == "true_positive" and r.correct
        )
        tp_total = sum(
            1 for r in report.results if r.original_label == "true_positive"
        )
        fp_correct = sum(
            1 for r in report.results
            if r.original_label == "false_positive" and r.correct
        )
        fp_total = sum(
            1 for r in report.results if r.original_label == "false_positive"
        )

        breakdown = Table(title="Per-class Accuracy")
        breakdown.add_column("Class", style="bold")
        breakdown.add_column("Correct / Total", justify="right")
        breakdown.add_column("Accuracy", justify="right")

        tp_acc = tp_correct / tp_total if tp_total else 0
        fp_acc = fp_correct / fp_total if fp_total else 0

        breakdown.add_row(
            "true_positive (target=1)",
            f"{tp_correct} / {tp_total}",
            f"{tp_acc:.1%}",
        )
        breakdown.add_row(
            "false_positive (target=0)",
            f"{fp_correct} / {fp_total}",
            f"{fp_acc:.1%}",
        )

        console.print(breakdown)


if __name__ == "__main__":
    app()
