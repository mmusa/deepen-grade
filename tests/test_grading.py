# SPDX-License-Identifier: Apache-2.0
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
