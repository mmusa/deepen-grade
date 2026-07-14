# SPDX-License-Identifier: Apache-2.0
"""Builds the `--json` report: a plain dict, CI-consumable, every check
result citing into a top-level `citations` map so nothing is duplicated and
nothing is uncited."""

from __future__ import annotations

from typing import Any

from deepen_grade.citations import ALL_CITATIONS
from deepen_grade.grading import DatasetGrade
from deepen_grade.report.badge import FUNNEL_ITEMS, SELF_ASSESSMENT_NOTE

# v2: calibration became its own top-level verdict (excluded from the score),
# and the honor-system "verified"/"badge_markdown" fields were removed --
# reports are explicitly local self-assessments now.
SCHEMA_VERSION = 2


def _result_to_dict(result) -> dict[str, Any]:
    return {
        "check_id": result.check_id,
        "name": result.name,
        "severity": result.severity.value,
        "summary": result.summary,
        "citations": list(result.citation_keys),
        "details": result.details,
    }


def build_report(grade: DatasetGrade) -> dict[str, Any]:
    used_citation_keys: set[str] = set()
    for result in grade.all_results():
        used_citation_keys.update(result.citation_keys)

    episodes = [
        {
            "episode_id": eg.episode_id,
            "score": eg.score,
            "grade": eg.letter,
            "task": eg.task,
            "success": eg.success,
            "duration_s": eg.duration_s,
            "checks": [_result_to_dict(r) for r in eg.results],
        }
        for eg in grade.episode_grades
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "deepen-grade",
        "report_type": "self-assessment",
        "self_assessment_note": SELF_ASSESSMENT_NOTE,
        "source": grade.source,
        "format": grade.format,
        "overall": {"score": grade.overall_score, "grade": grade.overall_letter},
        "calibration": {"verdict": grade.calibration_verdict, "detail": grade.calibration_detail},
        "dataset_level_checks": [_result_to_dict(r) for r in grade.dataset_level_results],
        "episodes": episodes,
        "citations": {
            key: {"title": c.title, "authors": c.authors, "venue": c.venue, "url": c.url}
            for key, c in ALL_CITATIONS.items()
            if key in used_citation_keys
        },
        "warnings": grade.warnings,
        "cannot_tell_you_locally": FUNNEL_ITEMS,
    }
