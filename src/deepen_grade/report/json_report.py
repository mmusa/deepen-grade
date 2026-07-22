# SPDX-License-Identifier: Apache-2.0
"""Builds the `--json` report: a plain dict, CI-consumable, every check
result citing into a top-level `citations` map so nothing is duplicated and
nothing is uncited."""

from __future__ import annotations

from typing import Any

from deepen_grade.citations import ALL_CITATIONS
from deepen_grade.grading import DatasetGrade
from deepen_grade.report.badge import FUNNEL_ITEMS, SELF_ASSESSMENT_NOTE

# v3: added "sampling" (present iff --sample/--max-episodes trimmed the
# episode set) and "partial" (true on an in-progress incremental write --
# see cli.py's --json -o path).
# v4: every check result now carries "claim_type" (DEFECT/RISK/CHARACTERISTIC/
# BY_DESIGN/NOT_ASSESSED -- see checks/base.py's ClaimType); the report grew a
# top-level "grade_schema" (grading.py's GRADE_SCHEMA, semver-versioned
# separately from this report envelope's own schema_version), a "defect_classes"
# breakdown (the DEFECT-only density behind "overall"), and a "training_value"
# profile (RISK-class findings, never counted in "overall"). Old consumers
# reading only "overall"/"episodes"/"checks" still work unchanged -- nothing
# existing was removed or renamed, which is what keeps this schema bump
# backward-comprehensible rather than a breaking rewrite.
SCHEMA_VERSION = 4


def _result_to_dict(result) -> dict[str, Any]:
    return {
        "check_id": result.check_id,
        "name": result.name,
        "severity": result.severity.value,
        "claim_type": result.claim_type.value,
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

    report = {
        "schema_version": SCHEMA_VERSION,
        "tool": "deepen-grade",
        "report_type": "self-assessment",
        "self_assessment_note": SELF_ASSESSMENT_NOTE,
        "grade_schema": grade.grade_schema,
        "partial": grade.partial,
        "source": grade.source,
        "format": grade.format,
        "overall": {"score": grade.overall_score, "grade": grade.overall_letter},
        "defect_classes": grade.defect_classes,
        "training_value": grade.training_value,
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
    if grade.sampling is not None:
        report["sampling"] = grade.sampling
    return report
