# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np

from deepen_grade.checks.base import CheckResult, ClaimType, Severity
from deepen_grade.grading import (
    CAL_NOT_ASSESSED,
    CAL_PRESENT_UNVERIFIED,
    CAL_STRUCTURALLY_BROKEN,
    CALIBRATION_CHECK_ID,
    GRADE_SCHEMA,
    LETTER_NOT_ASSESSED,
    grade_dataset,
    integrity_score,
    letter_from_score,
    score_from_results,
)
from deepen_grade.ingest.base import CalibrationInfo, Dataset, Episode, TopicInfo, Trajectory
from deepen_grade.report.json_report import build_report


def _result(severity, claim_type=ClaimType.DEFECT, check_id="x"):
    return CheckResult(check_id=check_id, name="x", severity=severity, summary="s", citation_keys=("rep105",),
                        claim_type=claim_type)


def test_score_from_results_perfect():
    assert score_from_results([_result(Severity.PASS)] * 5) == 100


def test_score_from_results_deducts():
    results = [_result(Severity.WARN), _result(Severity.FAIL)]
    assert score_from_results(results) == 100 - 4 - 12


def test_score_floors_at_zero():
    results = [_result(Severity.FAIL)] * 20
    assert score_from_results(results) == 0


def test_letter_bands():
    assert letter_from_score(100) == "A"
    assert letter_from_score(90) == "A"
    assert letter_from_score(89) == "B"
    assert letter_from_score(75) == "B"
    assert letter_from_score(60) == "C"
    assert letter_from_score(40) == "D"
    assert letter_from_score(0) == "F"


def _clean_episode(episode_id="e1"):
    n = 100
    t = np.arange(n) * 0.02
    state = np.stack([np.sin(t), np.cos(t)], axis=1)
    traj = Trajectory(timestamps_s=t, state=state, state_labels=None, action=state.copy(),
                       action_labels=None, gripper_position=None)
    cal = CalibrationInfo(camera_id="cam0", intrinsics={"fx": 500, "fy": 500, "cx": 320, "cy": 240},
                           extrinsics=[0.1, 0.2, 0.3, 0, 0, 1.57], source="test")
    return Episode(episode_id=episode_id, topics=[TopicInfo("/joint_states", "x", n, t)],
                    trajectory=traj, calibrations=[cal])


def test_grade_dataset_clean_episode_scores_well():
    ds = Dataset(source="test", format="lerobot", episodes=[_clean_episode()])
    grade = grade_dataset(ds)
    assert grade.overall_letter in ("A", "B")
    assert grade.episode_grades[0].episode_id == "e1"


def test_grade_dataset_empty():
    ds = Dataset(source="test", format="lerobot", episodes=[])
    grade = grade_dataset(ds)
    assert grade.overall_score == 0
    # Zero episodes is a special case of the NOT_ASSESSED floor (no DEFECT
    # check ever ran), not a graded F -- an empty gradable set must not
    # auto-fail any more than it auto-passes (QA finding 1).
    assert grade.overall_letter == LETTER_NOT_ASSESSED
    assert grade.episode_grades == []


def test_calibration_verdict_present_unverified():
    ds = Dataset(source="test", format="lerobot", episodes=[_clean_episode("a"), _clean_episode("b")])
    grade = grade_dataset(ds)
    assert grade.calibration_verdict == CAL_PRESENT_UNVERIFIED
    assert "cannot verify" in grade.calibration_detail  # spells out that accuracy is unverified


def test_calibration_verdict_not_assessed_when_no_metadata():
    ep = _clean_episode("a")
    ep.calibrations = []
    ds = Dataset(source="test", format="lerobot", episodes=[ep])
    grade = grade_dataset(ds)
    assert grade.calibration_verdict == CAL_NOT_ASSESSED


def test_calibration_verdict_structurally_broken():
    clean = _clean_episode("a")
    broken = _clean_episode("b")
    broken.calibrations = [CalibrationInfo(camera_id="cam0", intrinsics=None, extrinsics=None, source="x")]
    ds = Dataset(source="test", format="lerobot", episodes=[clean, broken])
    grade = grade_dataset(ds)
    assert grade.calibration_verdict == CAL_STRUCTURALLY_BROKEN


def test_calibration_never_moves_the_score():
    clean = _clean_episode("a")
    broken = _clean_episode("a")
    broken.calibrations = [CalibrationInfo(camera_id="cam0", intrinsics=None, extrinsics=None, source="x")]
    clean_grade = grade_dataset(Dataset(source="t", format="lerobot", episodes=[clean]))
    broken_grade = grade_dataset(Dataset(source="t", format="lerobot", episodes=[broken]))
    assert broken_grade.overall_score == clean_grade.overall_score
    # ...but the breakage is still visible on the check result itself.
    cal_results = [r for r in broken_grade.all_results() if r.check_id == CALIBRATION_CHECK_ID]
    assert any(r.severity == Severity.FAIL for r in cal_results)


# --- NOT_ASSESSED floor (QA finding 1: "vacuous A") --------------------------
#
# Root cause: topic_frequency_and_drops and tf_tree_integrity each abstain
# (claim_type NOT_ASSESSED) when their own preconditions aren't met -- no
# topics with enough messages, no tf concept -- and schema_and_dim_consistency
# abstains below 2 episodes. `_defect_density_by_check` filters NOT_ASSESSED
# results out entirely (never even creates a gradable=0 entry for them), so
# when EVERY DEFECT-eligible check abstains for a given dataset, `defect_classes`
# comes back completely empty and `_score_from_defect_classes`' `max(...,
# default=0.0)` silently reads that as "nothing failed" -> 100/A. This is
# distinct from a check running and PASSING (a real gradable=1, density=0.0
# entry) -- that's clean evidence and must still earn its A.


def _garbage_single_episode_dataset(episode_id: str) -> Dataset:
    """A single well-formed-but-content-garbage episode: valid, monotonic
    timestamps and a real (dead-constant) state array, but no `topics` and no
    `tf_edges` -- exactly the shape this repo's own plausibility tests build
    when they only need to exercise the trajectory surface. A single-episode
    Dataset is also exactly what every real .mcap/.bag file becomes (see
    ingest/mcap_reader.py, ingest/ros_common.py), so this is not a contrived
    shape -- it's what "grade one recording at a time" looks like for real.
    """
    n = 50
    t = np.arange(n) * 0.02
    garbage_state = np.zeros((n, 3))  # dead-constant -- not real robot data
    traj = Trajectory(timestamps_s=t, state=garbage_state, state_labels=None,
                       action=None, action_labels=None, gripper_position=None)
    ep = Episode(episode_id=episode_id, trajectory=traj)
    return Dataset(source=f"garbage/{episode_id}", format="lerobot", episodes=[ep])


def test_garbage_vacuous_dataset_is_not_assessed():
    ds = _garbage_single_episode_dataset("garbage_0")
    grade = grade_dataset(ds)
    assert grade.defect_classes == {}
    assert grade.overall_letter == LETTER_NOT_ASSESSED
    assert grade.overall_score == 0
    assert grade.plausibility_flagged is True
    assert "never change" in grade.plausibility_detail
    # The report must say WHY, not just issue a bare non-letter.
    assert any("NOT ASSESSED" in w and "gates that did not hold" in w.lower() for w in grade.warnings)


def test_qa_repro_exactly_five_garbage_episodes_each_not_assessed():
    """The literal QA finding 1 repro: 5 well-formed episodes of dead-constant
    garbage state, graded one at a time (as every real single-file source in
    this tool is) -- each must come back NOT ASSESSED, never a vacuous A."""
    for i in range(5):
        ds = _garbage_single_episode_dataset(f"garbage_{i}")
        grade = grade_dataset(ds)
        assert grade.defect_classes == {}, i
        assert grade.overall_letter == LETTER_NOT_ASSESSED, i
        assert grade.overall_score == 0, i
        assert grade.plausibility_flagged is True, i


def test_clean_dataset_with_passing_defect_checks_still_gets_an_a():
    """The floor's other half: DEFECT checks that actually RAN and PASSED
    (real gradable=1, density=0.0 evidence) must never be treated as absence
    of evidence -- only genuine silence triggers NOT_ASSESSED."""
    ds = Dataset(source="test", format="lerobot", episodes=[_clean_episode("a"), _clean_episode("b")])
    grade = grade_dataset(ds)
    assert grade.defect_classes  # non-empty: real DEFECT checks ran
    assert all(entry["gradable"] > 0 for entry in grade.defect_classes.values())
    assert grade.overall_letter == "A"
    assert grade.overall_score == 100
    assert grade.plausibility_flagged is False


def test_floor_does_not_trigger_merely_because_the_only_evidence_is_clean():
    """Multiple garbage episodes graded TOGETHER in one dataset can still give
    schema_and_dim_consistency real (if trivial) evidence to compare -- e.g.
    every episode's state genuinely has the same dim. That's a real DEFECT
    finding (density 0.0), not silence, so the floor must not override it even
    though the content is garbage (plausibility still flags it separately)."""
    n = 50
    t = np.arange(n) * 0.02
    episodes = []
    for i in range(3):
        traj = Trajectory(timestamps_s=t, state=np.zeros((n, 3)), state_labels=None,
                           action=None, action_labels=None, gripper_position=None)
        episodes.append(Episode(episode_id=f"garbage_{i}", trajectory=traj))
    ds = Dataset(source="garbage-multi", format="lerobot", episodes=episodes)
    grade = grade_dataset(ds)
    assert "hygiene.schema_dim_consistency" in grade.defect_classes
    assert grade.overall_letter != LETTER_NOT_ASSESSED
    assert grade.plausibility_flagged is True  # still surfaced, just doesn't force NOT_ASSESSED


def test_min_grade_gate_rejects_not_assessed_at_any_bar():
    from deepen_grade.cli import _LETTER_RANK

    assert LETTER_NOT_ASSESSED not in _LETTER_RANK


# --- incremental / partial grading (feature 4) -------------------------------


def test_on_progress_fires_every_batch_with_a_valid_partial_report():
    """Simulates a small --json -o batch by invoking the checkpoint function
    (grade_dataset's on_progress) directly with a tiny batch_size -- each
    callback must already be a fully parseable, "partial": true report, since
    that's exactly what the CLI atomically writes to disk mid-run."""
    episodes = [_clean_episode(f"e{i}") for i in range(7)]
    ds = Dataset(source="t", format="lerobot", episodes=episodes)

    seen: list[dict] = []

    def on_progress(partial_grade):
        report = build_report(partial_grade)
        json.dumps(report)  # must be serializable at every checkpoint
        seen.append(report)

    final = grade_dataset(ds, on_progress=on_progress, batch_size=3)

    # 7 episodes, batch_size=3 -> callbacks after episode 3 and episode 6 (not 7: that's the final return).
    assert [len(r["episodes"]) for r in seen] == [3, 6]
    assert all(r["partial"] is True for r in seen)

    final_report = build_report(final)
    assert final_report["partial"] is False
    assert len(final_report["episodes"]) == 7


def test_on_progress_interval_grows_with_progress():
    """Each flush rewrites the whole report, so a fixed 200-episode interval
    makes total bytes written quadratic in episode count (a 50k-episode run
    wrote ~100x its final report size). The interval must grow with progress:
    at 20k episodes a fixed interval would fire 99 times; the growing
    interval stays logarithmic, and each flush is at most ~10% of progress
    apart (the crash-loss bound)."""
    episodes = [_clean_episode(f"e{i}") for i in range(20_000)]
    ds = Dataset(source="t", format="lerobot", episodes=episodes)

    flush_points: list[int] = []

    def on_progress(partial_grade):
        flush_points.append(len(partial_grade.episode_grades))

    grade_dataset(ds, on_progress=on_progress)

    assert 5 < len(flush_points) < 40  # 99 would mean fixed-interval quadratic writes
    gaps = [b - a for a, b in zip(flush_points, flush_points[1:])]
    assert all(gap >= 200 for gap in gaps)
    for prev, cur in zip(flush_points, flush_points[1:]):
        assert cur - prev <= max(200, cur // 10) + 1  # loss bound holds


def test_grade_dataset_without_on_progress_never_calls_back():
    ds = Dataset(source="t", format="lerobot", episodes=[_clean_episode("a")])
    grade = grade_dataset(ds)  # default on_progress=None must not raise
    assert grade.partial is False


# --- claim-type CI guard: only DEFECT may ever move the letter --------------
#
# This introspects `integrity_score` (the actual aggregation function) with
# synthetic results built for every claim type, rather than spot-checking a
# real dataset -- a real dataset can accidentally "pass" this kind of test if
# no check happens to fire. Fails the build if any non-DEFECT claim type ever
# starts contributing.


def test_only_defect_class_touches_integrity_score():
    all_fail_defect = [_result(Severity.FAIL, ClaimType.DEFECT, check_id="defect_check")] * 50
    all_fail_risk = [_result(Severity.FAIL, ClaimType.RISK, check_id="risk_check")] * 50
    all_fail_characteristic = [_result(Severity.FAIL, ClaimType.CHARACTERISTIC, check_id="char_check")] * 50
    all_fail_by_design = [_result(Severity.FAIL, ClaimType.BY_DESIGN, check_id="by_design_check")] * 50
    all_fail_not_assessed = [_result(Severity.FAIL, ClaimType.NOT_ASSESSED, check_id="na_check")] * 50

    # A mountain of FAILs typed as anything but DEFECT must never move the score.
    assert integrity_score(all_fail_risk) == 100
    assert integrity_score(all_fail_characteristic) == 100
    assert integrity_score(all_fail_by_design) == 100
    assert integrity_score(all_fail_not_assessed) == 100
    assert integrity_score(all_fail_risk + all_fail_characteristic + all_fail_by_design + all_fail_not_assessed) == 100

    # The same FAILs typed DEFECT must tank the score -- proves the guard
    # above isn't just "integrity_score always returns 100."
    assert integrity_score(all_fail_defect) == 0

    # Mixing non-DEFECT noise in with a real DEFECT signal must not dilute it
    # (worst-class-bounded, not averaged) or hide it.
    mixed = all_fail_defect + all_fail_risk + all_fail_characteristic + all_fail_not_assessed
    assert integrity_score(mixed) == 0


def test_worst_class_bounded_not_averaged():
    """One fully-broken DEFECT class among clean ones must drive the score to
    0, not be averaged away by the clean classes -- GRADING_TAXONOMY_V1.md
    section 4.3's worked edge case."""
    broken_class = [_result(Severity.FAIL, ClaimType.DEFECT, check_id="broken")] * 10
    clean_class_a = [_result(Severity.PASS, ClaimType.DEFECT, check_id="clean_a")] * 10
    clean_class_b = [_result(Severity.PASS, ClaimType.DEFECT, check_id="clean_b")] * 10
    assert integrity_score(broken_class + clean_class_a + clean_class_b) == 0


def test_integrity_score_is_defect_density_not_a_single_fail_count():
    """A 20%-defective class scores 80, matching `100 * (1 - density)` --
    the exact functional form, not just "any FAIL drops the score to 0"."""
    results = (
        [_result(Severity.FAIL, ClaimType.DEFECT, check_id="c")] * 2
        + [_result(Severity.PASS, ClaimType.DEFECT, check_id="c")] * 8
    )
    assert integrity_score(results) == 80


def test_defect_classes_and_training_value_appear_on_the_dataset_grade():
    ds = Dataset(source="test", format="lerobot", episodes=[_clean_episode("a"), _clean_episode("b")])
    grade = grade_dataset(ds)
    assert grade.grade_schema == GRADE_SCHEMA
    assert "hygiene.topic_frequency" in grade.defect_classes
    # idle_stall etc. are RISK-typed and must show up in training_value, never in defect_classes.
    assert "episode_quality.idle_stall" in grade.training_value
    assert "episode_quality.idle_stall" not in grade.defect_classes
    for entry in grade.defect_classes.values():
        assert 0.0 <= entry["density"] <= 1.0

    report = build_report(grade)
    assert report["grade_schema"] == GRADE_SCHEMA
    assert report["defect_classes"] == grade.defect_classes
    assert report["training_value"] == grade.training_value
    json.dumps(report)  # both new sections must be JSON-serializable
