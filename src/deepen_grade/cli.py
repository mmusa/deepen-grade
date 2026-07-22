# SPDX-License-Identifier: Apache-2.0
"""`deepen grade <path-or-repo-id>` -- the CLI entry point."""

from __future__ import annotations

import json
import os
import sys

import click
from rich.console import Console
from rich.markup import escape

from deepen_grade.__about__ import __version__
from deepen_grade.grading import GRADE_BANDS, DatasetGrade, grade_dataset
from deepen_grade.ingest import load_dataset
from deepen_grade.ingest.exceptions import DatasetReadError, MissingOptionalDependencyError, UnsupportedFormatError
from deepen_grade.report import build_report, render_terminal, share_report
from deepen_grade.report.share import ShareNotImplementedError

_LETTER_RANK = {letter: i for i, (_, letter) in enumerate(GRADE_BANDS)}  # A=0 (best) .. F=4 (worst)


@click.group()
@click.version_option(__version__, prog_name="deepen-grade")
def main() -> None:
    """deepen-grade: grade robot-learning datasets for training-readiness, for free, locally.

    The letter grade is DEFECT-only (grade_schema 1.0.0): it reflects objective,
    gate-surviving defects (topic drops vs. a topic's own rate, a broken tf-tree,
    schema/dim mismatches), never a proxy training-quality signal. Idle, jerk,
    chatter, joint-limit, action-state, and cross-modal-skew findings are
    proxies, not correctness claims -- they're reported in the report's
    "training_value" section instead and never move the grade. See README's
    Grading section.
    """


@main.command()
@click.argument("source")
@click.option("--format", "fmt", type=click.Choice(["mcap", "ros1_bag", "ros2_db3", "lerobot"]),
              default=None, help="Skip auto-detection and force a format.")
@click.option("--json", "as_json", is_flag=True, help="Print the machine-readable JSON report instead of the terminal report.")
@click.option("-o", "--output", "output_path", type=click.Path(), default=None,
              help="Write the report to a file instead of stdout (JSON only). Written incrementally: "
                   "a partial report lands on disk periodically (at least every 10% of progress), so a crash/timeout mid-run "
                   "still leaves a valid, parseable file with \"partial\": true.")
@click.option("--min-grade", type=click.Choice(["A", "B", "C", "D", "F"]), default=None,
              help="Exit with a non-zero code if the overall grade is worse than this (for CI gating).")
@click.option("--cache-dir", type=click.Path(), default=None, help="Cache directory for downloaded HF repo-ids.")
@click.option("--subdataset", default=None,
              help="For a container repo-id/dir nesting multiple LeRobot datasets: grade only the one at this "
                   "relative path instead of all of them.")
@click.option("--sample", "sample", type=int, default=None,
              help="Grade a uniform random sample of N episodes instead of all of them (see --seed).")
@click.option("--seed", type=int, default=0, show_default=True, help="Random seed for --sample (deterministic).")
@click.option("--max-episodes", type=int, default=None,
              help="Grade only the first N episodes in natural order instead of all of them.")
@click.option("--hf-token", default=None, help="HuggingFace token for downloading a repo-id (or set HF_TOKEN).")
@click.option("--quiet", is_flag=True,
              help="Suppress HF download progress bars and CLI chatter that isn't part of the report.")
@click.option("--share-report", "share", is_flag=True,
              help="Opt-in: would POST the report to deepen-robograde.pages.dev. Not implemented yet in this release "
                   "(see --share-report-dry-run) -- nothing uploads by default.")
@click.option("--share-report-dry-run", "share_dry_run_path", type=click.Path(), default=None,
              help="With --share-report: write the payload that would be sent to this local file instead of failing bare.")
def grade(source, fmt, as_json, output_path, min_grade, cache_dir, subdataset, sample, seed, max_episodes,
          hf_token, quiet, share, share_dry_run_path) -> None:
    """Grade SOURCE: a .mcap/.bag/.db3 file, a LeRobot dataset directory, or a HuggingFace repo-id."""
    console = Console()
    err_console = Console(stderr=True)

    if quiet:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # 0 or negative silently graded "everything" (0 episodes) into a
    # confident-looking F(0/100) instead of erroring -- reject up front.
    if sample is not None and sample < 1:
        _fail(err_console, "--sample must be >= 1", exit_code=2, as_json=as_json)
    if max_episodes is not None and max_episodes < 1:
        _fail(err_console, "--max-episodes must be >= 1", exit_code=2, as_json=as_json)

    try:
        dataset = load_dataset(
            source, fmt=fmt, cache_dir=cache_dir, subdataset=subdataset, sample=sample,
            seed=seed, max_episodes=max_episodes, hf_token=hf_token,
        )
    except (MissingOptionalDependencyError, UnsupportedFormatError, DatasetReadError) as exc:
        _fail(err_console, str(exc), exit_code=2, as_json=as_json)

    on_progress = _make_progress_writer(output_path) if (as_json and output_path) else None
    dataset_grade = grade_dataset(dataset, on_progress=on_progress)
    report = build_report(dataset_grade)

    if share:
        try:
            share_report(report, dry_run_path=share_dry_run_path)
        except ShareNotImplementedError as exc:
            console.print(f"[yellow]{escape(str(exc))}[/yellow]")

    if as_json:
        text = json.dumps(report, indent=2)
        if output_path:
            _atomic_write(output_path, text)
        else:
            click.echo(text)
    else:
        render_terminal(dataset_grade, console=console)
        if output_path and not quiet:
            console.print("[dim]note: --output only writes a file with --json; ignored here.[/dim]")

    if min_grade is not None and _LETTER_RANK[dataset_grade.overall_letter] > _LETTER_RANK[min_grade]:
        sys.exit(1)
    sys.exit(0)


def _atomic_write(path: str, text: str) -> None:
    """Write-then-rename so a SIGTERM/timeout mid-write can never leave a
    truncated, unparseable file at `path` -- the last completed write (partial
    or final) is always what's on disk."""
    tmp_path = f"{path}.tmp-{os.getpid()}"
    with open(tmp_path, "w") as f:
        f.write(text)
    os.replace(tmp_path, path)


def _make_progress_writer(output_path: str):
    def _write_partial(partial_grade: DatasetGrade) -> None:
        _atomic_write(output_path, json.dumps(build_report(partial_grade), indent=2))

    return _write_partial


def _fail(err_console: Console, message: str, exit_code: int, as_json: bool) -> None:
    """Report a fatal error on stderr and exit. Errors always go to stderr,
    never stdout -- a pilot run once captured tqdm's stderr progress bars as
    "the output" because the actual error had gone to stdout instead. With
    --json, also emit a single-line JSON object on stderr (bypassing rich, so
    it's never word-wrapped) for automation to parse deterministically.
    """
    err_console.print(f"[bold red]error:[/bold red] {escape(message)}")
    if as_json:
        sys.stderr.write(json.dumps({"error": message, "exit_code": exit_code}) + "\n")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
