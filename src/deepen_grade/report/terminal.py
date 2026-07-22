# SPDX-License-Identifier: Apache-2.0
"""Human-readable terminal report, rendered with `rich`."""

from __future__ import annotations

from collections import Counter

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from deepen_grade.checks.base import Severity
from deepen_grade.citations import ALL_CITATIONS
from deepen_grade.grading import CAL_NOT_ASSESSED, CAL_PRESENT_UNVERIFIED, DatasetGrade
from deepen_grade.report.badge import SELF_ASSESSMENT_NOTE, funnel_text

SEVERITY_STYLE = {
    Severity.PASS: "green",
    Severity.INFO: "cyan",
    Severity.WARN: "yellow",
    Severity.FAIL: "bold red",
    Severity.NOT_APPLICABLE: "dim",
}
GRADE_STYLE = {"A": "bold green", "B": "green", "C": "yellow", "D": "bold yellow", "F": "bold red"}
CAL_VERDICT_STYLE = {
    CAL_NOT_ASSESSED: "bold yellow",
    CAL_PRESENT_UNVERIFIED: "cyan",
    # anything else (STRUCTURALLY BROKEN) renders bold red
}

# Above this many episodes, print per-check aggregates instead of a full
# per-episode check breakdown -- keeps large-dataset output readable.
MAX_EPISODES_FOR_DETAIL = 15


def _severity_text(severity: Severity) -> str:
    return f"[{SEVERITY_STYLE[severity]}]{severity.value.upper()}[/{SEVERITY_STYLE[severity]}]"


def _citation_text(citation_keys: tuple[str, ...]) -> str:
    if not citation_keys:
        return "-"
    return ", ".join(ALL_CITATIONS[k].venue for k in citation_keys if k in ALL_CITATIONS)


def render_terminal(grade: DatasetGrade, console: Console | None = None) -> None:
    # Dynamic, externally-sourced strings (dataset paths, task descriptions,
    # topic names embedded in check summaries, warnings) are passed through
    # `escape()` everywhere below before hitting a markup-parsing rich call --
    # a literal '[' in e.g. a HF repo-id or a task string like "grasp [red
    # block]" would otherwise be silently swallowed as (invalid) style markup.
    console = console or Console()

    grade_style = GRADE_STYLE.get(grade.overall_letter, "white")
    console.print(
        Panel(
            f"[bold]{escape(grade.source)}[/bold]\nformat: {grade.format}   "
            f"episodes: {len(grade.episode_grades)}",
            title="deepen-grade report",
        )
    )
    if grade.sampling is not None:
        s = grade.sampling
        seed_note = f", seed={s['seed']}" if s["mode"] == "sample" else ""
        console.print(
            f"[bold yellow]GRADED ON A SAMPLE[/bold yellow] ({s['n']} of {s['episodes_total']} "
            f"episodes, mode={s['mode']}{seed_note}) -- grade letters are unaffected but coverage isn't total."
        )
    if grade.partial:
        console.print("[bold yellow]PARTIAL REPORT[/bold yellow] -- grading was still in progress when this was written.")
    console.print(
        f"Overall grade: [{grade_style}]{grade.overall_letter}[/{grade_style}] "
        f"({grade.overall_score}/100)   [dim]grade_schema {escape(grade.grade_schema)} -- "
        "DEFECT-class findings only, see training_value below[/dim]"
    )
    # Calibration is a top-level verdict of its own, never part of the grade:
    # the grade must not imply calibration accuracy it cannot see.
    cal_style = CAL_VERDICT_STYLE.get(grade.calibration_verdict, "bold red")
    console.print(
        f"Calibration:   [{cal_style}]{grade.calibration_verdict}[/{cal_style}]\n"
        f"               [dim]{escape(grade.calibration_detail)}[/dim]"
    )

    for w in grade.warnings:
        console.print(f"[yellow]warning:[/yellow] {escape(w)}")

    if grade.dataset_level_results:
        table = Table(title="Dataset-level checks", show_lines=False)
        table.add_column("Check")
        table.add_column("Result")
        table.add_column("Summary")
        table.add_column("Citation")
        for r in grade.dataset_level_results:
            table.add_row(r.name, _severity_text(r.severity), escape(r.summary), _citation_text(r.citation_keys))
        console.print(table)

    ep_table = Table(title="Per-episode grades")
    ep_table.add_column("Episode")
    ep_table.add_column("Grade")
    ep_table.add_column("Score")
    ep_table.add_column("Task")
    for eg in grade.episode_grades:
        style = GRADE_STYLE.get(eg.letter, "white")
        ep_table.add_row(escape(eg.episode_id), f"[{style}]{eg.letter}[/{style}]", str(eg.score),
                          escape(eg.task) if eg.task else "-")
    console.print(ep_table)

    if grade.episode_grades:
        if len(grade.episode_grades) <= MAX_EPISODES_FOR_DETAIL:
            for eg in grade.episode_grades:
                console.print(_episode_detail_table(eg))
        else:
            console.print(_check_aggregate_table(grade))

    if grade.training_value:
        console.print(_training_value_table(grade.training_value))

    console.print(Panel(funnel_text(), title="Funnel", border_style="cyan"))
    console.print(Panel(Text(SELF_ASSESSMENT_NOTE), title="Self-assessment", border_style="yellow"))

    console.print("\nDeepen AI -- https://deepen.ai   |   Full audit: https://deepen-robograde.pages.dev")


def _episode_detail_table(eg) -> Table:
    table = Table(title=f"Checks -- {escape(eg.episode_id)}")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Summary")
    table.add_column("Citation")
    for r in eg.results:
        table.add_row(r.name, _severity_text(r.severity), escape(r.summary), _citation_text(r.citation_keys))
    return table


def _training_value_table(training_value: dict) -> Table:
    """RISK-class findings -- training-value proxy signals, never a letter
    input (see grading.py's `_training_value_profile`). Printed separately
    from the DEFECT-only grade so it's never mistaken for one.
    """
    table = Table(title="Training-value profile (RISK -- never graded)")
    table.add_column("Check")
    table.add_column("Pass")
    table.add_column("Warn")
    table.add_column("Fail")
    table.add_column("N/A")
    for entry in training_value.values():
        counts = entry["counts"]
        table.add_row(
            escape(entry["name"]),
            str(counts.get("pass", 0)),
            f"[yellow]{counts['warn']}[/yellow]" if counts.get("warn") else "0",
            f"[bold red]{counts['fail']}[/bold red]" if counts.get("fail") else "0",
            str(counts.get("n/a", 0)),
        )
    return table


def _check_aggregate_table(grade: DatasetGrade) -> Table:
    counts: dict[str, Counter] = {}
    names: dict[str, str] = {}
    for eg in grade.episode_grades:
        for r in eg.results:
            counts.setdefault(r.check_id, Counter())[r.severity] += 1
            names[r.check_id] = r.name

    table = Table(title=f"Check aggregate across {len(grade.episode_grades)} episodes")
    table.add_column("Check")
    table.add_column("Pass")
    table.add_column("Warn")
    table.add_column("Fail")
    table.add_column("N/A")
    for check_id, c in counts.items():
        table.add_row(
            names[check_id],
            str(c[Severity.PASS]),
            f"[yellow]{c[Severity.WARN]}[/yellow]" if c[Severity.WARN] else "0",
            f"[bold red]{c[Severity.FAIL]}[/bold red]" if c[Severity.FAIL] else "0",
            str(c[Severity.NOT_APPLICABLE]),
        )
    return table
