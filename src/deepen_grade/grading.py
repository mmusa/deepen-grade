# SPDX-License-Identifier: Apache-2.0
"""Aggregates check results into a per-episode grade and an overall dataset
grade.

GRADE_SCHEMA "1.0.0" (GRADING_TAXONOMY_V1.md, ratified 2026-07-21) is a
deliberate rewrite of how the letter is computed, not just a version bump:

  - The letter grade is now DEFECT-only. Every CheckResult carries a
    `claim_type` (see checks/base.py); only `ClaimType.DEFECT` findings can
    move the letter. RISK findings (idle/jerk/chatter/joint-limit/action-state
    consistency/cross-modal skew -- all proxy signals whose thresholds are
    conventions, not sourced standards) are reported in a separate
    `training_value` profile instead: numbers and proxy labels, zero letter
    impact. CHARACTERISTIC, BY_DESIGN, and NOT_ASSESSED findings never touch
    either.
  - The dataset's overall score is DEFECT-class defect DENSITY
    (defective/gradable), worst-class-bounded across DEFECT check classes --
    not the old "average every episode's penalty-sum score together." See
    `integrity_score`'s docstring for the exact functional form and why.

Anyone can still read this file and reproduce a grade by hand: group results
by check_id, compute each DEFECT check's density, take the worst, band it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from deepen_grade.checks import calibration_sanity, episode_quality, hygiene, plausibility
from deepen_grade.checks.base import CheckResult, ClaimType, Severity
from deepen_grade.ingest.base import Dataset, Episode

# After this many episodes are graded, an in-progress DatasetGrade is handed
# to `on_progress` (if given) so the caller can flush a partial report to disk
# -- a crash/timeout mid-run must still leave something readable behind
# instead of the zero-output-after-25-minutes failure this was built for.
PROGRESS_BATCH_SIZE = 200

# grade_schema carried on every report (json_report.py) -- see the module
# docstring's "Version migration" note and GRADING_TAXONOMY_V1.md section 4.6:
# a schema change is a semver bump with a documented "our method changed"
# rationale (README + CHANGELOG), never a silent renumbering.
GRADE_SCHEMA = "1.0.0"

SEVERITY_PENALTY: dict[Severity, int] = {
    Severity.PASS: 0,
    Severity.INFO: 0,
    Severity.NOT_APPLICABLE: 0,
    Severity.WARN: 4,
    Severity.FAIL: 12,
}

# (minimum score, letter) bands, checked highest-first. Applied to the 0-100
# integrity score from `integrity_score` (dataset level) or `score_from_results`
# (per-episode display score) -- both are on the same 0-100 scale, so one band
# table serves both. Kept as the pre-1.0.0 bands (no golden-corpus-derived
# percentile recalibration in this release -- see README's "Deviations" note).
GRADE_BANDS: tuple[tuple[int, str], ...] = ((90, "A"), (75, "B"), (60, "C"), (40, "D"), (0, "F"))

# Calibration NEVER counts toward the score. Local checks can only see whether
# calibration metadata is present and structurally sound -- a well-formed but
# geometrically wrong calibration passes all of them -- so blending this check
# into the grade would let a badly miscalibrated dataset grade well. It is
# reported instead as its own top-level verdict (see calibration_verdict).
# Excluded from both `_defect_density_by_check` and `score_penalty` by check_id,
# regardless of its own claim_type tag -- see calibration_sanity.py.
CALIBRATION_CHECK_ID = calibration_sanity.CHECK_ID

# Dataset-level calibration verdict values.
CAL_NOT_ASSESSED = "NOT ASSESSED"
CAL_PRESENT_UNVERIFIED = "PRESENT -- ACCURACY NOT VERIFIED"
CAL_STRUCTURALLY_BROKEN = "STRUCTURALLY BROKEN"

# The letter when NO DEFECT-class check produced even one gradable finding
# anywhere in the dataset -- see `_assemble_grade`'s floor. Deliberately not a
# GRADE_BANDS entry: it must never rank alongside A-F (an "empty gradable set"
# is a different kind of outcome than "graded and clean," per
# GRADING_TAXONOMY_V1.md section 3 -- "never auto-passes... never auto-fails").
LETTER_NOT_ASSESSED = "NOT ASSESSED"

PLAUSIBILITY_CHECK_ID = plausibility.CHECK_ID


def score_penalty(results: list[CheckResult]) -> int:
    """Fixed-point WARN/FAIL penalty, DEFECT-class findings only. Used for the
    per-episode display score (`score_from_results`) -- never for the dataset's
    overall letter, which is `integrity_score`'s defect density instead.
    """
    return sum(
        SEVERITY_PENALTY[r.severity]
        for r in results
        if r.claim_type == ClaimType.DEFECT and r.check_id != CALIBRATION_CHECK_ID
    )


def score_from_results(results: list[CheckResult]) -> int:
    return max(0, 100 - score_penalty(results))


def letter_from_score(score: int) -> str:
    for threshold, letter in GRADE_BANDS:
        if score >= threshold:
            return letter
    return "F"  # pragma: no cover -- unreachable, GRADE_BANDS floors at 0


def _defect_density_by_check(results: list[CheckResult]) -> dict[str, dict]:
    """Group DEFECT-claim results by check_id and compute each check's defect
    density: defective findings (severity WARN or FAIL) over gradable findings
    (anything but NOT_APPLICABLE). A check_id with zero gradable findings
    abstains (density 0.0) rather than counting as either a pass or a fail --
    it simply can't move the worst-class bound in `integrity_score`.

    This is the ONLY place that decides what's letter-eligible, and it decides
    purely from `claim_type`, never from a parallel check_id allowlist -- see
    the CI guard test (test_grading.py) that constructs synthetic results of
    every claim type and asserts only DEFECT moves the score.
    """
    by_check: dict[str, dict] = {}
    for r in results:
        if r.claim_type != ClaimType.DEFECT or r.check_id == CALIBRATION_CHECK_ID:
            continue
        entry = by_check.setdefault(r.check_id, {"name": r.name, "gradable": 0, "defective": 0})
        if r.severity == Severity.NOT_APPLICABLE:
            continue
        entry["gradable"] += 1
        if r.severity in (Severity.WARN, Severity.FAIL):
            entry["defective"] += 1
    for entry in by_check.values():
        entry["density"] = entry["defective"] / entry["gradable"] if entry["gradable"] else 0.0
    return by_check


def _score_from_defect_classes(by_check: dict[str, dict]) -> int:
    """`100 * (1 - worst_density)`, rounded and clamped to [0, 100] -- the
    arithmetic behind `integrity_score`, factored out so `_assemble_grade` can
    reuse the SAME `by_check` dict it already needs for the report's
    `defect_classes` field instead of recomputing it from scratch.
    """
    worst_density = max((c["density"] for c in by_check.values()), default=0.0)
    return max(0, min(100, round(100 * (1 - worst_density))))


def _has_defect_evidence(by_check: dict[str, dict]) -> bool:
    """True iff at least one DEFECT-class check produced at least one gradable
    (non-N/A) finding SOMEWHERE in the dataset. False is the "empty gradable
    set" case `_assemble_grade`'s NOT_ASSESSED floor exists for: every DEFECT
    check individually abstained (no topics to measure frequency on, no tf
    concept, fewer than 2 episodes to compare schemas across, ...), so
    `_defect_density_by_check` never even created an entry for them --
    `integrity_score` would otherwise read that silence as "nothing failed"
    and hand back a perfect, unearned 100.
    """
    return any(entry["gradable"] > 0 for entry in by_check.values())


def integrity_score(results: list[CheckResult]) -> int:
    """The dataset's letter-grade input: DEFECT-only, worst-class-bounded
    defect density (Deepen convention v1, GRADING_TAXONOMY_V1.md section 4.3).

    Functional form: `100 * (1 - worst_density)`, rounded and clamped to
    [0, 100], where `worst_density` is the MAXIMUM (not average) defect
    density across every DEFECT-class check_id (see `_defect_density_by_check`).
    Worked edge cases:
      - One systemic defect class among five clean ones (e.g. tf-tree broken
        in every episode, everything else perfect): worst_density = 1.0 for
        that class alone => score = 0, regardless of the other four classes.
        An average would have diluted this to ~80/100 -- exactly the "one bad
        class averaged away" failure this rule exists to prevent.
      - Five classes all at a mediocre 20% density: worst_density = 0.20 =>
        score = 80, same as if only one class were at 20% and the rest were
        perfect. This is a deliberate, published tradeoff (the rejected
        alternative -- weighted mean with a floor -- was considered and
        dropped because it still lets enough small-but-widespread badness
        average toward a passing grade; worst-class-bounded never does).

    RISK/CHARACTERISTIC/BY_DESIGN/NOT_ASSESSED findings never enter this
    function's inputs (filtered by claim_type in `_defect_density_by_check`),
    so no non-DEFECT finding can move this number by construction.

    NOTE: this returns a plain 0-100 number even when NO DEFECT check ever
    produced gradable evidence at all (density then defaults to 0.0 => 100).
    `_assemble_grade` is what actually issues the letter, and it checks
    `_has_defect_evidence` FIRST and overrides to `LETTER_NOT_ASSESSED` before
    ever consulting this score for that case -- see its docstring.
    """
    return _score_from_defect_classes(_defect_density_by_check(results))


def _not_assessed_floor_warning(results: list[CheckResult]) -> str:
    """Explains the NOT_ASSESSED letter by naming every gate that didn't hold
    (Law 1: "Couldn't measure," "measured fine," and "measured bad" are three
    different outputs -- silence must never be reported as a number). Lists
    every NOT_ASSESSED finding's check_id and its own summary, deduplicated,
    not just the DEFECT-eligible ones -- a reader trying to understand "why
    can't this be graded" benefits from the full picture (missing tf concept,
    too few episodes, no topics to measure a rate on, ...), not a subset.
    """
    reasons: dict[str, str] = {}
    for r in results:
        if r.claim_type == ClaimType.NOT_ASSESSED and r.check_id not in reasons:
            reasons[r.check_id] = r.summary
    gate_list = "; ".join(f"{check_id} ({summary})" for check_id, summary in sorted(reasons.items()))
    return (
        "NOT ASSESSED: no DEFECT-class check produced a single gradable finding anywhere "
        "in this dataset, so no letter grade can be issued -- an empty gradable set must "
        "never auto-pass (GRADING_TAXONOMY_V1.md section 3). "
        f"Gates that did not hold: {gate_list or '(none named)'}."
    )


def _plausibility_flag(dataset_results: list[CheckResult]) -> tuple[bool, str | None]:
    """Whether `plausibility.robot_data` fired on this dataset, and its
    summary. Surfaced as its own top-level field (report + terminal), not left
    buried in the dataset-level checks table: GRADING_TAXONOMY_V1.md's class F
    treats this as an index-curation gate ("does this even look like robot
    data"), and a confident-looking letter grade must never silently coexist
    with "this doesn't look like robot-learning data" -- see `_assemble_grade`.
    """
    for r in dataset_results:
        if r.check_id == PLAUSIBILITY_CHECK_ID and r.severity == Severity.INFO:
            return True, r.summary
    return False, None


def _training_value_profile(results: list[CheckResult]) -> dict[str, dict]:
    """RISK-class findings, summarized per check_id: severity counts and
    nothing else -- the "training_value" report section. These are proxy
    training-quality signals (idle fraction, LDJ smoothness, gripper chatter,
    joint-limit saturation, action/state tracking, cross-modal skew), useful
    for curating a training set, never a correctness claim -- see
    GRADING_TAXONOMY_V1.md section 2C/3 for why their thresholds don't qualify
    as DEFECT-grade evidence. Never read by `integrity_score`.
    """
    by_check: dict[str, dict] = {}
    for r in results:
        if r.claim_type != ClaimType.RISK:
            continue
        entry = by_check.setdefault(r.check_id, {"name": r.name, "counts": {}})
        entry["counts"][r.severity.value] = entry["counts"].get(r.severity.value, 0) + 1
    return by_check


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
    grade_schema: str = GRADE_SCHEMA
    defect_classes: dict = field(default_factory=dict)  # per DEFECT check_id: see _defect_density_by_check
    training_value: dict = field(default_factory=dict)  # per RISK check_id: see _training_value_profile
    plausibility_flagged: bool = False  # see _plausibility_flag
    plausibility_detail: str | None = None

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


def grade_episode(episode: Episode, dim_stats: dict[int, episode_quality.DimStats] | None = None) -> EpisodeGrade:
    results = [
        *hygiene.run_episode_checks(episode),
        *episode_quality.run_checks(episode, dim_stats),
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
    all_results = dataset_results + [r for eg in episode_grades for r in eg.results]
    defect_classes = _defect_density_by_check(all_results)

    # The NOT_ASSESSED floor (GRADING_TAXONOMY_V1.md section 3: "an empty
    # gradable set never auto-passes"). This is NOT "no episodes" specifically
    # -- it's the general case where every DEFECT-class check individually
    # abstained (no topics to measure a rate on, no tf concept, fewer than 2
    # episodes to compare schemas across, ...), so `defect_classes` came back
    # with no gradable entries anywhere. `_score_from_defect_classes` would
    # silently read that as "nothing failed" (density 0.0 => 100). A clean
    # dataset whose DEFECT checks actually RAN and PASSED is untouched by this
    # -- `_has_defect_evidence` is true the moment even one check produced a
    # single gradable finding, gradable=1/density=0.0 included.
    if _has_defect_evidence(defect_classes):
        overall_score = _score_from_defect_classes(defect_classes)
        overall_letter = letter_from_score(overall_score)
        warnings = list(dataset.warnings)
    else:
        overall_score = 0
        overall_letter = LETTER_NOT_ASSESSED
        warnings = [*dataset.warnings, _not_assessed_floor_warning(all_results)]

    verdict, detail = calibration_verdict(episode_grades)
    plausibility_flagged, plausibility_detail = _plausibility_flag(dataset_results)

    return DatasetGrade(
        source=dataset.source,
        format=dataset.format,
        episode_grades=list(episode_grades),
        dataset_level_results=dataset_results,
        overall_score=overall_score,
        overall_letter=overall_letter,
        warnings=warnings,
        calibration_verdict=verdict,
        calibration_detail=detail,
        sampling=dataset.sampling,
        partial=partial,
        defect_classes=defect_classes,
        training_value=_training_value_profile(all_results),
        plausibility_flagged=plausibility_flagged,
        plausibility_detail=plausibility_detail,
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
    progress callback too. The dataset-wide per-dim range/discreteness stats
    that `joint_limit_saturation` needs (never a per-episode range -- see
    episode_quality.py) are computed here for the same reason: every episode
    is already materialized before grading starts, so this is the first point
    a true dataset-wide aggregate is available.
    """
    dataset_results = hygiene.run_dataset_checks(dataset) + plausibility.run_checks(dataset)
    dim_stats = episode_quality.dataset_dim_stats(dataset.episodes)
    episode_grades: list[EpisodeGrade] = []
    last_flush = 0
    for i, ep in enumerate(dataset.episodes, start=1):
        episode_grades.append(grade_episode(ep, dim_stats))
        if (
            on_progress is not None
            and i < len(dataset.episodes)
            and i - last_flush >= max(batch_size, i // 10)
        ):
            on_progress(_assemble_grade(dataset, episode_grades, dataset_results, partial=True))
            last_flush = i

    return _assemble_grade(dataset, episode_grades, dataset_results, partial=False)
