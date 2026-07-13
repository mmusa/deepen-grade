# SPDX-License-Identifier: Apache-2.0
"""Aggregates check results into a per-episode grade and an overall dataset
grade. The formula is deliberately simple and fully documented here (not a
sealed/tuned model): start at 100, deduct fixed points per WARN/FAIL, map the
resulting score onto a letter band. Anyone can read this file and reproduce
a grade by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from deepen_grade.checks import calibration_sanity, episode_quality, hygiene
from deepen_grade.checks.base import CheckResult, Severity
from deepen_grade.ingest.base import Dataset, Episode

SEVERITY_PENALTY: dict[Severity, int] = {
    Severity.PASS: 0,
    Severity.INFO: 0,
    Severity.NOT_APPLICABLE: 0,
    Severity.WARN: 4,
    Severity.FAIL: 12,
}

# (minimum score, letter) bands, checked highest-first.
GRADE_BANDS: tuple[tuple[int, str], ...] = ((90, "A"), (75, "B"), (60, "C"), (40, "D"), (0, "F"))


def score_penalty(results: list[CheckResult]) -> int:
    return sum(SEVERITY_PENALTY[r.severity] for r in results)


def score_from_results(results: list[CheckResult]) -> int:
    return max(0, 100 - score_penalty(results))


def letter_from_score(score: int) -> str:
    for threshold, letter in GRADE_BANDS:
        if score >= threshold:
            return letter
    return "F"  # pragma: no cover -- unreachable, GRADE_BANDS floors at 0


@dataclass
class EpisodeGrade:
    episode_id: str
    score: int
    letter: str
    results: list[CheckResult]
    task: str | None = None
    success: bool | None = None
    duration_s: float | None = None


@dataclass
class DatasetGrade:
    source: str
    format: str
    episode_grades: list[EpisodeGrade]
    dataset_level_results: list[CheckResult] = field(default_factory=list)
    overall_score: int = 0
    overall_letter: str = "F"
    warnings: list[str] = field(default_factory=list)

    def all_results(self) -> list[CheckResult]:
        return self.dataset_level_results + [r for eg in self.episode_grades for r in eg.results]


def grade_episode(episode: Episode) -> EpisodeGrade:
    results = [
        *hygiene.run_episode_checks(episode),
        *episode_quality.run_checks(episode),
        *calibration_sanity.run_checks(episode),
    ]
    score = score_from_results(results)
    return EpisodeGrade(
        episode_id=episode.episode_id,
        score=score,
        letter=letter_from_score(score),
        results=results,
        task=episode.task,
        success=episode.success,
        duration_s=episode.duration_s,
    )


def grade_dataset(dataset: Dataset) -> DatasetGrade:
    episode_grades = [grade_episode(ep) for ep in dataset.episodes]
    dataset_results = hygiene.run_dataset_checks(dataset)

    base_score = int(round(np.mean([g.score for g in episode_grades]))) if episode_grades else 0
    overall_score = max(0, base_score - score_penalty(dataset_results))

    return DatasetGrade(
        source=dataset.source,
        format=dataset.format,
        episode_grades=episode_grades,
        dataset_level_results=dataset_results,
        overall_score=overall_score,
        overall_letter=letter_from_score(overall_score),
        warnings=list(dataset.warnings),
    )


# "Deepen Verified" badge eligibility -- transparent, documented rule: a B or
# better overall AND no FAIL anywhere (dataset-level or in any episode).
def passes_verification(grade: DatasetGrade) -> bool:
    if grade.overall_letter not in ("A", "B"):
        return False
    return not any(r.severity == Severity.FAIL for r in grade.all_results())
