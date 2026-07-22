# SPDX-License-Identifier: Apache-2.0
"""Regression test: rich's markup parser silently swallows literal '[...]'
text (e.g. a HF repo-id, a task description, or our own 'pip install
"deepen-grade[mcap]"' hint) unless it's escaped before being handed to a
markup-parsing rich call. See report/terminal.py's `escape()` usage."""

from rich.console import Console

from deepen_grade.checks.base import CheckResult, ClaimType, Severity
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
    result = CheckResult(check_id="x", name="x", severity=Severity.PASS, summary="s", citation_keys=("rep105",),
                          claim_type=ClaimType.DEFECT)
    eg = EpisodeGrade(episode_id="e1", score=90, letter="A", results=[result],
                       task="grasp [red block] and place in [bin 2]")
    grade = DatasetGrade(source="test", format="lerobot", episode_grades=[eg])
    output = _console_output(grade)
    assert "grasp [red block] and place in [bin 2]" in output


def test_check_summary_with_brackets_survives_rendering():
    result = CheckResult(
        check_id="x", name="x", severity=Severity.WARN,
        summary="topic [weird/name] flagged", citation_keys=("rep105",),
        claim_type=ClaimType.DEFECT,
    )
    eg = EpisodeGrade(episode_id="e1", score=90, letter="B", results=[result])
    grade = DatasetGrade(source="test", format="lerobot", episode_grades=[eg])
    output = _console_output(grade)
    assert "topic [weird/name] flagged" in output


# --- QA finding 2: RISK findings must be visually distinct from DEFECT ones --


def test_risk_fail_is_labeled_distinctly_from_defect_fail():
    """A RISK-class FAIL (e.g. gripper chatter) sitting right under "Overall
    grade: A" must be visibly marked as non-letter-affecting -- a reader
    scanning the table must be able to tell it apart from a DEFECT FAIL that
    actually earned that grade."""
    defect_fail = CheckResult(check_id="d", name="Defect check", severity=Severity.FAIL,
                               summary="s", citation_keys=("rep105",), claim_type=ClaimType.DEFECT)
    risk_fail = CheckResult(check_id="r", name="Risk check", severity=Severity.FAIL,
                             summary="s", citation_keys=("rep105",), claim_type=ClaimType.RISK)
    eg = EpisodeGrade(episode_id="e1", score=88, letter="B", results=[defect_fail, risk_fail])
    grade = DatasetGrade(source="test", format="lerobot", episode_grades=[eg])
    output = _console_output(grade)
    assert "defect" in output.lower()
    assert "risk" in output.lower()
    # A footnote spelling out that only DEFECT rows move the letter.
    assert "only defect" in output.lower() or "never do" in output.lower()


def test_content_plausibility_flag_prints_prominently_next_to_grade():
    grade = DatasetGrade(
        source="garbage", format="lerobot", episode_grades=[],
        plausibility_flagged=True, plausibility_detail="state values never change across any sampled episode",
    )
    output = _console_output(grade)
    assert "CONTENT PLAUSIBILITY FLAG" in output
    assert "state values never change" in output
