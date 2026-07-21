# SPDX-License-Identifier: Apache-2.0
"""Calibration SANITY check -- and *only* sanity.

Per deepen-grade's IP firewall (see README "What this can't tell you
locally"): this module checks whether intrinsics/extrinsics are *present* and
*not obviously broken* (missing, NaN, an untouched identity/zero default).

It does **not** verify that a calibration is geometrically correct. Doing that
from a recording alone (reprojection/epipolar/triangulation analysis, drift
detection over a session) is Deepen's patented targetless calibration
verification -- sealed, server-side, offered at deepen-robograde.pages.dev. Nothing in
this file should ever grow toward that; if you're tempted to add a residual,
an epipolar distance, or anything that opens a video frame, stop -- that
belongs in the paid audit, not here.
"""

from __future__ import annotations

import numpy as np

from deepen_grade.checks.base import CheckResult, Severity
from deepen_grade.citations import SHADOW_EXTRINSICS_NOISE
from deepen_grade.ingest.base import CalibrationInfo, Episode

FUNNEL_NOTE = (
    "deepen-grade only checks that calibration is present and not obviously "
    "broken. It cannot tell you whether the numbers are geometrically correct "
    "-- run the full targetless calibration audit at deepen-robograde.pages.dev."
)

# Calibration is reported as its own top-level verdict and NEVER blended into
# the letter grade (grading.py excludes this check_id from scoring): a
# well-formed but geometrically wrong calibration passes every local check, so
# letting this check move the score would let a badly miscalibrated dataset
# grade well -- exactly the false confidence the grade must not sell.
CHECK_ID = "calibration.sanity"


def _intrinsics_issue(intrinsics: dict | None) -> str | None:
    if intrinsics is None:
        return "intrinsics missing"
    values = [intrinsics.get(k) for k in ("fx", "fy", "cx", "cy")]
    if any(v is None for v in values):
        return "intrinsics incomplete (missing fx/fy/cx/cy)"
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).any():
        return "intrinsics contain NaN"
    if values[0] == 0 or values[1] == 0:  # fx or fy == 0 => degenerate projection
        return "intrinsics degenerate (fx or fy is zero)"
    return None


def _is_identity_or_zero(vec: np.ndarray) -> bool:
    if vec.size == 16:
        return bool(np.allclose(vec.reshape(4, 4), np.eye(4), atol=1e-6))
    return bool(np.allclose(vec, 0.0, atol=1e-9))


def _extrinsics_issue(extrinsics: list[float] | None) -> str | None:
    if extrinsics is None:
        return "extrinsics missing"
    arr = np.asarray(extrinsics, dtype=float)
    if np.isnan(arr).any():
        return "extrinsics contain NaN"
    if _is_identity_or_zero(arr):
        return "extrinsics look like an unset identity/zero default"
    return None


def _check_one(cal: CalibrationInfo) -> list[str]:
    issues = []
    for issue in (_intrinsics_issue(cal.intrinsics), _extrinsics_issue(cal.extrinsics)):
        if issue:
            issues.append(issue)
    return issues


def calibration_sanity(episode: Episode) -> CheckResult:
    if not episode.calibrations:
        return CheckResult(
            check_id=CHECK_ID,
            name="Calibration sanity",
            severity=Severity.INFO,
            summary="NOT ASSESSED -- no camera calibration metadata found for this episode",
            citation_keys=(SHADOW_EXTRINSICS_NOISE.key,),
            details={"cameras": [], "funnel": FUNNEL_NOTE},
        )

    flagged = {}
    for cal in episode.calibrations:
        issues = _check_one(cal)
        if issues:
            flagged[cal.camera_id] = issues

    if flagged:
        severity = Severity.FAIL
        summary = (
            f"STRUCTURALLY BROKEN -- {len(flagged)}/{len(episode.calibrations)} "
            f"camera(s) flagged: {', '.join(flagged)}"
        )
    else:
        severity = Severity.PASS
        summary = (
            f"PRESENT -- {len(episode.calibrations)} camera(s): intrinsics/extrinsics "
            "structurally sound (presence check only -- accuracy NOT verified)"
        )

    return CheckResult(
        check_id=CHECK_ID,
        name="Calibration sanity",
        severity=severity,
        summary=summary,
        citation_keys=(SHADOW_EXTRINSICS_NOISE.key,),
        details={"cameras": [cal.camera_id for cal in episode.calibrations],
                 "flagged": flagged, "funnel": FUNNEL_NOTE},
    )


def run_checks(episode: Episode) -> list[CheckResult]:
    return [calibration_sanity(episode)]
