# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np

from deepen_grade.checks.base import CheckResult, Severity
from deepen_grade.grading import (
    CAL_NOT_ASSESSED,
    CAL_PRESENT_UNVERIFIED,
    CAL_STRUCTURALLY_BROKEN,
    CALIBRATION_CHECK_ID,
    grade_dataset,
    letter_from_score,
    score_from_results,
)
from deepen_grade.ingest.base import CalibrationInfo, Dataset, Episode, TopicInfo, Trajectory
from deepen_grade.report.json_report import build_report


def _result(severity):
    return CheckResult(check_id="x", name="x", severity=severity, summary="s", citation_keys=("rep105",))


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
