# SPDX-License-Identifier: Apache-2.0
"""Aggregates check results into a per-episode grade and an overall dataset
grade. The formula is deliberately simple and fully documented here (not a
sealed/tuned model): start at 100, deduct fixed points per WARN/FAIL, map the
resulting score onto a letter band. Anyone can read this file and reproduce
a grade by hand.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from deepen_grade.checks import calibration_sanity, episode_quality, hygiene, plausibility
from deepen_grade.checks.base import CheckResult, Severity
from deepen_grade.ingest.base import Dataset, Episode

# After this many episodes are graded, an in-progress DatasetGrade is handed
# to `on_progress` (if given) so the caller can flush a partial report to disk
# -- a crash/timeout mid-run must still leave something readable behind
# instead of the zero-output-after-25-minutes failure this was built for.
PROGRESS_BATCH_SIZE = 200

SEVERITY_PENALTY: dict[Severity, int] = {
    Severity.PASS: 0,
    Severity.INFO: 0,
    Severity.NOT_APPLICABLE: 0,
    Severity.WARN: 4,
    Severity.FAIL: 12,
}

# (minimum score, letter) bands, checked highest-first.
GRADE_BANDS: tuple[tuple[int, str], ...] = ((90, "A"), (75, "B"), (60, "C"), (40, "D"), (0, "F"))

# Calibration NEVER counts toward the score. Local checks can only see whether
# calibration metadata is present and structurally sound -- a well-formed but
# geometrically wrong calibration passes all of them -- so blending this check
# into the grade would let a badly miscalibrated dataset grade well. It is
# reported instead as its own top-level verdict (see calibration_verdict).
CALIBRATION_CHECK_ID = calibration_sanity.CHECK_ID

# Dataset-level calibration verdict values.
CAL_NOT_ASSESSED = "NOT ASSESSED"
CAL_PRESENT_UNVERIFIED = "PRESENT -- ACCURACY NOT VERIFIED"
CAL_STRUCTURALLY_BROKEN = "STRUCTURALLY BROKEN"


def score_penalty(results: list[CheckResult]) -> int:
    return sum(SEVERITY_PENALTY[r.severity] for r in results if r.check_id != CALIBRATION_CHECK_ID)


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
    calibration_verdict: str = CAL_NOT_ASSESSED
    calibration_detail: str = ""
    sampling: dict | None = None  # see Dataset.sampling
    partial: bool = False  # True on an in-progress report handed to on_progress

    def all_results(self) -> list[CheckResult]:
        return self.dataset_level_results + [r for eg in self.episode_grades for r in eg.results]


def calibration_verdict(episode_grades: list[EpisodeGrade]) -> tuple[str, str]:
    """The top-level calibration verdict, kept OUT of the letter grade."""
    cal_results = [r for eg in episode_grades for r in eg.results if r.check_id == CALIBRATION_CHECK_ID]
    broken = sum(1 for r in cal_results if r.severity == Severity.FAIL)
    present = sum(1 for r in cal_results if r.severity == Severity.PASS)

    if broken:
        return CAL_STRUCTURALLY_BROKEN, (
            f"calibration metadata is present but obviously broken (missing/NaN/unset default) "
            f"in {broken}/{len(cal_results)} episode(s) -- fix before training or trading on this data"
        )
    if present:
        return CAL_PRESENT_UNVERIFIED, (
            f"calibration metadata present in {present}/{len(cal_results)} episode(s) and structurally "
            "sound, but deepen-grade cannot verify the numbers are geometrically correct -- "
            "that requires the deep audit"
        )
    return CAL_NOT_ASSESSED, (
        "no calibration metadata found anywhere in this dataset -- calibration quality is "
        "unknown, not good; verification requires the deep audit"
    )


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


def _assemble_grade(
    dataset: Dataset, episode_grades: list[EpisodeGrade], dataset_results: list[CheckResult], partial: bool
) -> DatasetGrade:
    base_score = int(round(np.mean([g.score for g in episode_grades]))) if episode_grades else 0
    overall_score = max(0, base_score - score_penalty(dataset_results))
    verdict, detail = calibration_verdict(episode_grades)

    return DatasetGrade(
        source=dataset.source,
        format=dataset.format,
        episode_grades=list(episode_grades),
        dataset_level_results=dataset_results,
        overall_score=overall_score,
        overall_letter=letter_from_score(overall_score),
        warnings=list(dataset.warnings),
        calibration_verdict=verdict,
        calibration_detail=detail,
        sampling=dataset.sampling,
        partial=partial,
    )


def grade_dataset(
    dataset: Dataset,
    on_progress: Callable[[DatasetGrade], None] | None = None,
    batch_size: int = PROGRESS_BATCH_SIZE,
) -> DatasetGrade:
    """Grade every episode plus the dataset-level checks.

    `on_progress`, if given, is called with an in-progress (`partial=True`)
    DatasetGrade periodically -- the CLI's `--json -o` incremental-write path
    uses this to flush a valid report to disk instead of only at the end.
    Each flush rewrites the whole report, so a fixed interval makes total
    bytes written grow quadratically with episode count (a 50k-episode run
    wrote ~100x its final report size). The interval therefore grows with
    progress: at least `batch_size` episodes apart, and at least 10% of the
    episodes graded so far -- total write volume stays within ~10x the final
    report size while a crash still loses at most ~10% of progress.
    Dataset-level checks (schema/dim consistency, plausibility) only need
    `dataset.episodes`' metadata, which is already fully materialized by the
    time grading starts, so they run once up front and are included in every
    progress callback too.
    """
    dataset_results = hygiene.run_dataset_checks(dataset) + plausibility.run_checks(dataset)
    episode_grades: list[EpisodeGrade] = []
    last_flush = 0
    for i, ep in enumerate(dataset.episodes, start=1):
        episode_grades.append(grade_episode(ep))
        if (
            on_progress is not None
            and i < len(dataset.episodes)
            and i - last_flush >= max(batch_size, i // 10)
        ):
            on_progress(_assemble_grade(dataset, episode_grades, dataset_results, partial=True))
            last_flush = i

    return _assemble_grade(dataset, episode_grades, dataset_results, partial=False)
