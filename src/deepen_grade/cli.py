# SPDX-License-Identifier: Apache-2.0
"""`deepen grade <path-or-repo-id>` -- the CLI entry point."""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.markup import escape

from deepen_grade.__about__ import __version__
from deepen_grade.grading import GRADE_BANDS, grade_dataset
from deepen_grade.ingest import load_dataset
from deepen_grade.ingest.exceptions import DatasetReadError, MissingOptionalDependencyError, UnsupportedFormatError
from deepen_grade.report import build_report, render_terminal, share_report
from deepen_grade.report.share import ShareNotImplementedError

_LETTER_RANK = {letter: i for i, (_, letter) in enumerate(GRADE_BANDS)}  # A=0 (best) .. F=4 (worst)


@click.group()
@click.version_option(__version__, prog_name="deepen-grade")
def main() -> None:
    """deepen-grade: grade robot-learning datasets for training-readiness, for free, locally."""


@main.command()
@click.argument("source")
@click.option("--format", "fmt", type=click.Choice(["mcap", "ros1_bag", "ros2_db3", "lerobot"]),
              default=None, help="Skip auto-detection and force a format.")
@click.option("--json", "as_json", is_flag=True, help="Print the machine-readable JSON report instead of the terminal report.")
@click.option("-o", "--output", "output_path", type=click.Path(), default=None,
              help="Write the report to a file instead of stdout (JSON only).")
@click.option("--min-grade", type=click.Choice(["A", "B", "C", "D", "F"]), default=None,
              help="Exit with a non-zero code if the overall grade is worse than this (for CI gating).")
@click.option("--cache-dir", type=click.Path(), default=None, help="Cache directory for downloaded HF repo-ids.")
@click.option("--share-report", "share", is_flag=True,
              help="Opt-in: would POST the report to evaluate.deepen.ai. Not implemented yet in this release "
                   "(see --share-report-dry-run) -- nothing uploads by default.")
@click.option("--share-report-dry-run", "share_dry_run_path", type=click.Path(), default=None,
              help="With --share-report: write the payload that would be sent to this local file instead of failing bare.")
def grade(source, fmt, as_json, output_path, min_grade, cache_dir, share, share_dry_run_path) -> None:
    """Grade SOURCE: a .mcap/.bag/.db3 file, a LeRobot dataset directory, or a HuggingFace repo-id."""
    console = Console()
    try:
        dataset = load_dataset(source, fmt=fmt, cache_dir=cache_dir)
    except (MissingOptionalDependencyError, UnsupportedFormatError, DatasetReadError) as exc:
        console.print(f"[bold red]error:[/bold red] {escape(str(exc))}")
        sys.exit(2)

    dataset_grade = grade_dataset(dataset)
    report = build_report(dataset_grade)

    if share:
        try:
            share_report(report, dry_run_path=share_dry_run_path)
        except ShareNotImplementedError as exc:
            console.print(f"[yellow]{escape(str(exc))}[/yellow]")

    if as_json:
        text = json.dumps(report, indent=2)
        if output_path:
            with open(output_path, "w") as f:
                f.write(text)
        else:
            click.echo(text)
    else:
        render_terminal(dataset_grade, console=console)
        if output_path:
            console.print("[dim]note: --output only writes a file with --json; ignored here.[/dim]")

    if min_grade is not None and _LETTER_RANK[dataset_grade.overall_letter] > _LETTER_RANK[min_grade]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
