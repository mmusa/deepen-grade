# SPDX-License-Identifier: Apache-2.0
"""Regression test: rich's markup parser silently swallows literal '[...]'
text (e.g. a HF repo-id, a task description, or our own 'pip install
"deepen-grade[mcap]"' hint) unless it's escaped before being handed to a
markup-parsing rich call. See report/terminal.py's `escape()` usage."""

from rich.console import Console

from deepen_grade.checks.base import CheckResult, Severity
from deepen_grade.grading import DatasetGrade, EpisodeGrade
from deepen_grade.report.terminal import render_terminal


def _console_output(grade: DatasetGrade) -> str:
    console = Console(record=True, width=120)
    render_terminal(grade, console=console)
    return console.export_text()


def test_dataset_warning_with_brackets_survives_rendering():
    grade = DatasetGrade(
        source="some/repo[mcap]", format="mcap", episode_grades=[],
        warnings=['install with `pip install "deepen-grade[mcap]"`'],
    )
    output = _console_output(grade)
    assert "deepen-grade[mcap]" in output
    assert "some/repo[mcap]" in output


def test_episode_task_with_brackets_survives_rendering():
    result = CheckResult(check_id="x", name="x", severity=Severity.PASS, summary="s", citation_keys=("rep105",))
    eg = EpisodeGrade(episode_id="e1", score=90, letter="A", results=[result],
                       task="grasp [red block] and place in [bin 2]")
    grade = DatasetGrade(source="test", format="lerobot", episode_grades=[eg])
    output = _console_output(grade)
    assert "grasp [red block] and place in [bin 2]" in output


def test_check_summary_with_brackets_survives_rendering():
    result = CheckResult(
        check_id="x", name="x", severity=Severity.WARN,
        summary="topic [weird/name] flagged", citation_keys=("rep105",),
    )
    eg = EpisodeGrade(episode_id="e1", score=90, letter="B", results=[result])
    grade = DatasetGrade(source="test", format="lerobot", episode_grades=[eg])
    output = _console_output(grade)
    assert "topic [weird/name] flagged" in output
