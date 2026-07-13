# SPDX-License-Identifier: Apache-2.0
"""Episode-quality checks: idle/stall, movement smoothness, joint-limit
saturation, gripper chatter, and action-state consistency.

Metric definitions are grounded in two published sources:
  - DQAF (Siemens, arXiv:2605.26349) names exactly this metric suite --
    stalling, joint-limit proximity, gripper chatter, smoothness -- as the
    published episode-level teleop quality framework.
  - SCIZOR (arXiv:2505.22626) independently curates on idle/redundant
    state-action pairs at OXE scale.
  - Log-dimensionless-jerk itself is Balasubramanian et al. 2015 (the metric
    DQAF applies to teleop episodes).

IMPORTANT on thresholds: the cited papers define *what to measure*. The
pass/warn/fail *cutoffs* below (e.g. "stall_frac > 0.30 is a FAIL") are
deepen-grade's own conservative, documented engineering defaults -- not
values taken from the papers -- because a universal numeric cutoff across
every robot embodiment and control rate doesn't exist in the literature.
They are intentionally simple and stated here in one place so anyone can
audit or override them; see README for rationale.
"""

from __future__ import annotations

import numpy as np

from deepen_grade.checks.base import CheckResult, Severity
from deepen_grade.citations import DQAF, LDJ_SMOOTHNESS, SCIZOR
from deepen_grade.ingest.base import Episode, Trajectory

MIN_SAMPLES = 6

# --- idle / stall -----------------------------------------------------------
IDLE_SPEED_FRAC = 0.005      # normalized speed below this counts as "idle"
STALL_WARN_FRAC = 0.10       # fraction of episode duration spent idle
STALL_FAIL_FRAC = 0.30

# --- joint-limit saturation --------------------------------------------------
SATURATION_MARGIN_FRAC = 0.02   # within this fraction of observed range = "at the limit"
SATURATION_MIN_RUN_S = 0.2      # must be sustained this long to count (not sensor noise)
SATURATION_WARN_FRAC = 0.10
SATURATION_FAIL_FRAC = 0.25

# --- gripper chatter ----------------------------------------------------------
CHATTER_WARN_HZ = 2.0
CHATTER_FAIL_HZ = 5.0

# --- smoothness (log-dimensionless-jerk) -------------------------------------
LDJ_WARN = -20.0
LDJ_FAIL = -40.0

# --- action/state consistency -------------------------------------------------
CONSISTENCY_WARN = 0.8
CONSISTENCY_FAIL = 0.5


def _median_dt(t: np.ndarray) -> float | None:
    if len(t) < 2:
        return None
    dt = float(np.median(np.diff(t)))
    return dt if dt > 0 else None


def _normalized_speed(pos: np.ndarray, dt: float) -> np.ndarray:
    """Per-dim min-max normalized velocity magnitude -- dimensionless so
    joints measured in radians and grippers in meters contribute comparably.
    """
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    rng = np.maximum(hi - lo, 1e-9)
    normed = (pos - lo) / rng
    vel = np.gradient(normed, dt, axis=0)
    return np.linalg.norm(vel, axis=1)


def _longest_run_frac(mask: np.ndarray, min_run: int) -> tuple[float, int]:
    """Fraction of True samples that belong to a run of length >= min_run, and
    the longest run length."""
    if not mask.any():
        return 0.0, 0
    covered = 0
    longest = 0
    run = 0
    for v in mask:
        if v:
            run += 1
        else:
            if run >= min_run:
                covered += run
            longest = max(longest, run)
            run = 0
    if run >= min_run:
        covered += run
    longest = max(longest, run)
    return covered / len(mask), longest


def idle_stall(trajectory: Trajectory) -> CheckResult:
    pos = trajectory.action if trajectory.action is not None else trajectory.state
    if pos is None or len(trajectory.timestamps_s) < MIN_SAMPLES:
        return _na("episode_quality.idle_stall", "Idle / stall detection")

    dt = _median_dt(trajectory.timestamps_s)
    if dt is None:
        return _na("episode_quality.idle_stall", "Idle / stall detection")

    speed = _normalized_speed(pos, dt)
    idle_mask = speed < IDLE_SPEED_FRAC
    stall_frac, longest_run = _longest_run_frac(idle_mask, min_run=max(3, int(round(0.5 / dt))))
    longest_s = longest_run * dt
    duration = float(trajectory.timestamps_s[-1] - trajectory.timestamps_s[0])

    if stall_frac >= STALL_FAIL_FRAC:
        severity = Severity.FAIL
    elif stall_frac >= STALL_WARN_FRAC:
        severity = Severity.WARN
    else:
        severity = Severity.PASS
    summary = f"{stall_frac * 100:.1f}% of episode idle (longest stall {longest_s:.1f}s of {duration:.1f}s)"

    return CheckResult(
        check_id="episode_quality.idle_stall",
        name="Idle / stall detection",
        severity=severity,
        summary=summary,
        citation_keys=(DQAF.key, SCIZOR.key),
        details={"stall_frac": round(stall_frac, 4), "longest_stall_s": round(longest_s, 2),
                 "duration_s": round(duration, 2)},
    )


def jerk_smoothness(trajectory: Trajectory) -> CheckResult:
    pos = trajectory.state if trajectory.state is not None else trajectory.action
    if pos is None or len(trajectory.timestamps_s) < MIN_SAMPLES:
        return _na("episode_quality.jerk_smoothness", "Log-dimensionless-jerk smoothness")

    dt = _median_dt(trajectory.timestamps_s)
    ldj = _log_dimensionless_jerk(pos, dt) if dt else None
    if ldj is None:
        return _na("episode_quality.jerk_smoothness", "Log-dimensionless-jerk smoothness")

    if ldj <= LDJ_FAIL:
        severity = Severity.FAIL
    elif ldj <= LDJ_WARN:
        severity = Severity.WARN
    else:
        severity = Severity.PASS

    return CheckResult(
        check_id="episode_quality.jerk_smoothness",
        name="Log-dimensionless-jerk smoothness",
        severity=severity,
        summary=f"LDJ = {ldj:.1f} (higher/less-negative = smoother)",
        citation_keys=(LDJ_SMOOTHNESS.key, DQAF.key),
        details={"ldj": round(ldj, 2)},
    )


def _log_dimensionless_jerk(pos: np.ndarray, dt: float) -> float | None:
    """Balasubramanian et al. 2015, applied to the speed profile of a
    (possibly multi-dimensional) trajectory. Assumes roughly uniform sampling
    (true for fixed-rate robot control loops) -- `dt` is the trajectory's
    median sample interval.
    """
    vel = np.gradient(pos, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    peak_speed = float(speed.max())
    if peak_speed <= 1e-9:
        return None
    accel = np.gradient(speed, dt)
    jerk = np.gradient(accel, dt)
    duration = float(len(pos) - 1) * dt
    integral_jerk_sq = float(np.sum(jerk**2) * dt)
    scale = (duration**3) / (peak_speed**2)
    value = scale * integral_jerk_sq
    if value <= 0:
        return None
    return float(-np.log(value))


def joint_limit_saturation(trajectory: Trajectory) -> CheckResult:
    state = trajectory.state
    if state is None or len(trajectory.timestamps_s) < MIN_SAMPLES:
        return _na("episode_quality.joint_limit_saturation", "Joint-limit saturation")

    dt = _median_dt(trajectory.timestamps_s)
    if dt is None:
        return _na("episode_quality.joint_limit_saturation", "Joint-limit saturation")
    min_run = max(3, int(round(SATURATION_MIN_RUN_S / dt)))

    labels = trajectory.state_labels
    per_dim: dict[str, float] = {}
    for d in range(state.shape[1]):
        col = state[:, d]
        lo, hi = float(col.min()), float(col.max())
        rng = hi - lo
        if rng <= 1e-9:
            continue
        near_edge = (col <= lo + SATURATION_MARGIN_FRAC * rng) | (col >= hi - SATURATION_MARGIN_FRAC * rng)
        frac, _ = _longest_run_frac(near_edge, min_run)
        label = labels[d] if labels and d < len(labels) else f"dim_{d}"
        per_dim[label] = round(frac, 4)

    if not per_dim:
        return _na("episode_quality.joint_limit_saturation", "Joint-limit saturation")

    worst_dim = max(per_dim, key=per_dim.get)
    worst_frac = per_dim[worst_dim]
    if worst_frac >= SATURATION_FAIL_FRAC:
        severity = Severity.FAIL
    elif worst_frac >= SATURATION_WARN_FRAC:
        severity = Severity.WARN
    else:
        severity = Severity.PASS

    summary = (
        f"worst dim '{worst_dim}' saturated {worst_frac * 100:.1f}% of episode "
        f"(observed-range proxy, no declared joint limits available)"
    )
    return CheckResult(
        check_id="episode_quality.joint_limit_saturation",
        name="Joint-limit saturation",
        severity=severity,
        summary=summary,
        citation_keys=(DQAF.key,),
        details={"per_dim_saturation_frac": per_dim, "worst_dim": worst_dim},
    )


CHATTER_WINDOW_S = 1.0  # local burst window; a whole-episode average would dilute a short burst away


def gripper_chatter(trajectory: Trajectory) -> CheckResult:
    grip = trajectory.gripper_position
    if grip is None or len(grip) < MIN_SAMPLES:
        return _na("episode_quality.gripper_chatter", "Gripper chatter")

    t = trajectory.timestamps_s[: len(grip)]
    dt = _median_dt(t)
    if dt is None:
        return _na("episode_quality.gripper_chatter", "Gripper chatter")

    vel = np.gradient(grip, dt)
    sign = np.sign(vel)
    sign[sign == 0] = 1
    reversal_mask = np.abs(np.diff(sign)) > 0
    reversal_times = t[:-1][reversal_mask]
    total_reversals = int(reversal_mask.sum())

    # Peak local rate over a sliding window, not the whole-episode average --
    # a short chatter burst in an otherwise-smooth episode must not get
    # diluted into invisibility by the episode's total duration.
    if total_reversals == 0:
        peak_rate_hz = 0.0
    else:
        counts = [
            int(np.sum((reversal_times >= t0) & (reversal_times < t0 + CHATTER_WINDOW_S)))
            for t0 in reversal_times
        ]
        peak_rate_hz = max(counts) / CHATTER_WINDOW_S

    if peak_rate_hz >= CHATTER_FAIL_HZ:
        severity = Severity.FAIL
    elif peak_rate_hz >= CHATTER_WARN_HZ:
        severity = Severity.WARN
    else:
        severity = Severity.PASS

    return CheckResult(
        check_id="episode_quality.gripper_chatter",
        name="Gripper chatter",
        severity=severity,
        summary=f"{total_reversals} direction reversals total, peak {peak_rate_hz:.1f} Hz "
                f"over a {CHATTER_WINDOW_S:g}s window",
        citation_keys=(DQAF.key,),
        details={"reversals": total_reversals, "peak_rate_hz": round(peak_rate_hz, 3)},
    )


def action_state_consistency(trajectory: Trajectory) -> CheckResult:
    state, action = trajectory.state, trajectory.action
    if state is None or action is None:
        return _na("episode_quality.action_state_consistency", "Action-state consistency")

    if state.shape[1] != action.shape[1]:
        return CheckResult(
            check_id="episode_quality.action_state_consistency",
            name="Action-state consistency",
            severity=Severity.FAIL,
            summary=f"action dim ({action.shape[1]}) != state dim ({state.shape[1]})",
            citation_keys=(DQAF.key, SCIZOR.key),
            details={"state_dim": int(state.shape[1]), "action_dim": int(action.shape[1])},
        )

    n = min(len(state), len(action))
    state, action = state[:n], action[:n]
    rng = np.maximum(state.max(axis=0) - state.min(axis=0), 1e-9)
    norm_abs_err = np.abs(action - state) / rng
    consistency = float(np.clip(1.0 - norm_abs_err.mean(), 0.0, 1.0))

    if consistency < CONSISTENCY_FAIL:
        severity = Severity.FAIL
    elif consistency < CONSISTENCY_WARN:
        severity = Severity.WARN
    else:
        severity = Severity.PASS

    return CheckResult(
        check_id="episode_quality.action_state_consistency",
        name="Action-state consistency",
        severity=severity,
        summary=f"consistency score {consistency:.2f} (1.0 = action always matches resulting state)",
        citation_keys=(DQAF.key, SCIZOR.key),
        details={"consistency_score": round(consistency, 4)},
    )


def _na(check_id: str, name: str) -> CheckResult:
    return CheckResult(
        check_id=check_id, name=name, severity=Severity.NOT_APPLICABLE,
        summary="insufficient decodable state/action data for this episode",
        citation_keys=(),
    )


def run_checks(episode: Episode) -> list[CheckResult]:
    if episode.trajectory is None:
        return [
            _na("episode_quality.idle_stall", "Idle / stall detection"),
            _na("episode_quality.jerk_smoothness", "Log-dimensionless-jerk smoothness"),
            _na("episode_quality.joint_limit_saturation", "Joint-limit saturation"),
            _na("episode_quality.gripper_chatter", "Gripper chatter"),
            _na("episode_quality.action_state_consistency", "Action-state consistency"),
        ]
    t = episode.trajectory
    return [
        idle_stall(t),
        jerk_smoothness(t),
        joint_limit_saturation(t),
        gripper_chatter(t),
        action_state_consistency(t),
    ]
