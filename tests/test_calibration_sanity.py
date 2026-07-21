# SPDX-License-Identifier: Apache-2.0
from deepen_grade.checks import calibration_sanity as cs
from deepen_grade.checks.base import Severity
from deepen_grade.ingest.base import CalibrationInfo, Episode

GOOD_INTRINSICS = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
GOOD_EXTRINSICS = [0.1, 0.2, 0.3, 0.0, 0.0, 1.57]


def test_no_calibration_present_is_not_assessed_and_funnels():
    ep = Episode(episode_id="e1", calibrations=[])
    result = cs.calibration_sanity(ep)
    assert result.severity == Severity.INFO
    assert "NOT ASSESSED" in result.summary
    assert "deepen-robograde.pages.dev" in result.details["funnel"]


def test_valid_calibration_passes():
    cal = CalibrationInfo(camera_id="cam0", intrinsics=GOOD_INTRINSICS, extrinsics=GOOD_EXTRINSICS, source="test")
    ep = Episode(episode_id="e1", calibrations=[cal])
    result = cs.calibration_sanity(ep)
    assert result.severity == Severity.PASS


def test_nan_extrinsics_fails():
    cal = CalibrationInfo(camera_id="cam0", intrinsics=GOOD_INTRINSICS,
                           extrinsics=[float("nan"), 0, 0, 0, 0, 0], source="test")
    ep = Episode(episode_id="e1", calibrations=[cal])
    result = cs.calibration_sanity(ep)
    assert result.severity == Severity.FAIL
    assert "cam0" in result.details["flagged"]
    assert any("NaN" in issue for issue in result.details["flagged"]["cam0"])


def test_identity_extrinsics_flagged():
    cal = CalibrationInfo(camera_id="cam0", intrinsics=GOOD_INTRINSICS,
                           extrinsics=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], source="test")
    ep = Episode(episode_id="e1", calibrations=[cal])
    result = cs.calibration_sanity(ep)
    assert result.severity == Severity.FAIL
    assert any("identity/zero" in issue for issue in result.details["flagged"]["cam0"])


def test_missing_intrinsics_flagged():
    cal = CalibrationInfo(camera_id="cam0", intrinsics=None, extrinsics=GOOD_EXTRINSICS, source="test")
    ep = Episode(episode_id="e1", calibrations=[cal])
    result = cs.calibration_sanity(ep)
    assert result.severity == Severity.FAIL
    assert "intrinsics missing" in result.details["flagged"]["cam0"]


def test_zero_focal_length_degenerate():
    bad_intrinsics = {"fx": 0.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
    cal = CalibrationInfo(camera_id="cam0", intrinsics=bad_intrinsics, extrinsics=GOOD_EXTRINSICS, source="test")
    ep = Episode(episode_id="e1", calibrations=[cal])
    result = cs.calibration_sanity(ep)
    assert result.severity == Severity.FAIL
