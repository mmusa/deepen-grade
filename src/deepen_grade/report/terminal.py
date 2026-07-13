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
from deepen_grade.grading import DatasetGrade
from deepen_grade.report.badge import badge_markdown, funnel_text

SEVERITY_STYLE = {
    Severity.PASS: "green",
    Severity.INFO: "cyan",
    Severity.WARN: "yellow",
    Severity.FAIL: "bold red",
    Severity.NOT_APPLICABLE: "dim",
}
GRADE_STYLE = {"A": "bold green", "B": "green", "C": "yellow", "D": "bold yellow", "F": "bold red"}

# Above this many episodes, print per-check aggregates instead of a full
# per-episode check breakdown -- keeps large-dataset output readable.
MAX_EPISODES_FOR_DETAIL = 15


def _severity_text(severity: Severity) -> str:
    return f"[{SEVERITY_STYLE[severity]}]{severity.value.upper()}[/{SEVERITY_STYLE[severity]}]"


def _citation_text(citation_keys: tuple[str, ...]) -> str:
    if not citation_keys:
        return "-"
    return ", ".join(ALL_CITATIONS[k].venue for k in citation_keys if k in ALL_CITATIONS)


def render_terminal(grade: DatasetGrade, verified: bool, console: Console | None = None) -> None:
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
    console.print(
        f"Overall grade: [{grade_style}]{grade.overall_letter}[/{grade_style}] "
        f"({grade.overall_score}/100)"
        + ("   [bold green]Deepen Verified eligible[/bold green]" if verified else "")
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

    console.print(Panel(funnel_text(), title="Funnel", border_style="cyan"))

    if verified:
        # Text(), not a markup string: this is the literal snippet a user
        # copies into their dataset card -- it must render byte-for-byte,
        # not be parsed as rich markup.
        console.print(Panel(Text(badge_markdown(grade.source)),
                             title="Deepen Verified badge (paste into your dataset card)"))

    console.print("\nDeepen AI -- https://deepen.ai   |   Full audit: https://evaluate.deepen.ai")


def _episode_detail_table(eg) -> Table:
    table = Table(title=f"Checks -- {escape(eg.episode_id)}")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Summary")
    table.add_column("Citation")
    for r in eg.results:
        table.add_row(r.name, _severity_text(r.severity), escape(r.summary), _citation_text(r.citation_keys))
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
