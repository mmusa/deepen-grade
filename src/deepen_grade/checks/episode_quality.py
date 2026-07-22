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

CLAIM TYPES (GRADING_TAXONOMY_V1.md section 2C -- "kinematic & dynamic
plausibility", ceiling RISK): every check in this module is a training-value
proxy, never a correctness claim, and is typed `ClaimType.RISK` accordingly --
none of them can move the letter grade (see grading.py). idle/jerk/chatter are
proxies within their stated validity domain by construction (there's no
"declared idle budget" or "declared limit" metadata to gate them against
today), so they're RISK unconditionally. `action_state_consistency` and
`joint_limit_saturation` carry an extra NOT_ASSESSED / CHARACTERISTIC path
each -- see their docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepen_grade.checks.base import CheckResult, ClaimType, Severity
from deepen_grade.citations import DQAF, LDJ_SMOOTHNESS, SCIZOR
from deepen_grade.ingest.base import Episode, Trajectory

MIN_SAMPLES = 6

# --- idle / stall -----------------------------------------------------------
IDLE_SPEED_FRAC = 0.005      # normalized speed below this counts as "idle"
STALL_WARN_FRAC = 0.10       # fraction of episode duration spent idle
STALL_FAIL_FRAC = 0.30

# --- joint-limit saturation --------------------------------------------------
SATURATION_MARGIN_FRAC = 0.02   # within this fraction of the dataset-wide range = "at the limit"
SATURATION_MIN_RUN_S = 0.2      # must be sustained this long to count (not sensor noise)
SATURATION_WARN_FRAC = 0.10
SATURATION_FAIL_FRAC = 0.25
# A dim with fewer distinct values than this across the WHOLE dataset is
# discrete/binary (a 0/1 gripper flag, a mode enum), not a continuous joint --
# every sample of a binary dim sits "at the limit" by construction, so the
# saturation proxy is meaningless for it and must be excluded rather than
# reporting a permanent, content-free "100% saturated" (the exact bug this
# check shipped with: see GRADING_TAXONOMY_V1.md's origin story).
SATURATION_MIN_UNIQUE_VALUES = 5
# A dim whose dataset-wide (hi - lo) is below this has no meaningful excursion
# to saturate at all (already excluded below independent of the unique-value
# count -- kept as its own constant since "near-constant" and "discrete" are
# different failure modes, per GRADING_TAXONOMY_V1.md section 2C).
SATURATION_MIN_RANGE = 1e-9

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
    the longest run length.

    Vectorized run-length encoding (edges via np.diff) instead of a per-sample
    Python loop: this is called once per episode for idle/stall and once per
    *state dimension* for joint-limit saturation, so an O(n) Python loop here
    multiplies badly across a large episode count (see README perf note).
    """
    if not mask.any():
        return 0.0, 0
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    run_lengths = edges[1::2] - edges[0::2]  # [start, end) pairs -> lengths
    longest = int(run_lengths.max())
    covered = int(run_lengths[run_lengths >= min_run].sum())
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
        claim_type=ClaimType.RISK,
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
        claim_type=ClaimType.RISK,
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


@dataclass
class DimStats:
    """Dataset-wide statistics for one state dim, aggregated once over every
    episode before any episode's joint_limit_saturation runs (see
    `dataset_dim_stats`). Never computed per-episode: a short precision
    primitive that only visits a fraction of a joint's real range would
    otherwise saturate its own local excursion by construction -- the exact
    bug GRADING_TAXONOMY_V1.md's origin story names.
    """

    lo: float
    hi: float
    is_discrete: bool  # fewer than SATURATION_MIN_UNIQUE_VALUES distinct values dataset-wide


def dataset_dim_stats(episodes: list[Episode]) -> dict[int, DimStats]:
    """Aggregate per-state-dim (lo, hi, is_discrete) across every episode's
    trajectory in the dataset. `is_discrete` only needs to know whether a dim's
    distinct-value count clears `SATURATION_MIN_UNIQUE_VALUES` at all -- once
    the running set does, the dim is confirmed continuous and is never grown
    further, so a genuinely continuous dim only costs real work for the
    handful of episodes it takes to trip the cap (almost always the first
    one), keeping this close to O(dataset size) without ever holding a full
    corpus-wide value set.

    Each episode's FULL distinct-value set is unioned in (never truncated to
    e.g. its lowest few values first): a per-episode truncation is what
    originally let a genuinely continuous joint that repeatedly revisits the
    same low "reset/home" values across episodes get misclassified as
    discrete -- the higher-valued excursions later in each episode were being
    sliced off before they ever reached the running set, so the set never
    grew past the cap even though the joint's true range was wide.
    """
    lo: dict[int, float] = {}
    hi: dict[int, float] = {}
    uniques: dict[int, set[float]] = {}

    for ep in episodes:
        traj = ep.trajectory
        if traj is None or traj.state is None or len(traj.state) == 0:
            continue
        for d in range(traj.state.shape[1]):
            col = traj.state[:, d]
            col_lo, col_hi = float(col.min()), float(col.max())
            lo[d] = min(lo.get(d, col_lo), col_lo)
            hi[d] = max(hi.get(d, col_hi), col_hi)
            seen = uniques.setdefault(d, set())
            if len(seen) <= SATURATION_MIN_UNIQUE_VALUES:
                seen.update(np.unique(np.round(col, 9)).tolist())

    return {
        d: DimStats(lo=lo[d], hi=hi[d], is_discrete=len(uniques[d]) <= SATURATION_MIN_UNIQUE_VALUES)
        for d in lo
    }


def joint_limit_saturation(trajectory: Trajectory, dim_stats: dict[int, DimStats]) -> CheckResult:
    """Time spent near each dim's DATASET-WIDE observed range -- never the
    episode's own range (see `dataset_dim_stats`). Binary/near-constant dims
    (a 0/1 gripper flag) are excluded outright: every sample of a binary dim
    sits "at the limit" by construction, which is a proxy artifact, not a
    finding. No declared joint limits exist anywhere in this codebase's data
    model, so this is always a RISK-class training-value signal, never a
    DEFECT -- see the module docstring.
    """
    state = trajectory.state
    if state is None or len(trajectory.timestamps_s) < MIN_SAMPLES:
        return _na("episode_quality.joint_limit_saturation", "Joint-limit saturation")

    dt = _median_dt(trajectory.timestamps_s)
    if dt is None:
        return _na("episode_quality.joint_limit_saturation", "Joint-limit saturation")
    min_run = max(3, int(round(SATURATION_MIN_RUN_S / dt)))

    labels = trajectory.state_labels
    per_dim: dict[str, float] = {}
    excluded_discrete: list[str] = []
    for d in range(state.shape[1]):
        stats = dim_stats.get(d)
        label = labels[d] if labels and d < len(labels) else f"dim_{d}"
        if stats is None:
            continue
        rng = stats.hi - stats.lo
        if rng <= SATURATION_MIN_RANGE:
            continue
        if stats.is_discrete:
            excluded_discrete.append(label)
            continue
        col = state[:, d]
        near_edge = (
            (col <= stats.lo + SATURATION_MARGIN_FRAC * rng) | (col >= stats.hi - SATURATION_MARGIN_FRAC * rng)
        )
        frac, _ = _longest_run_frac(near_edge, min_run)
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
        f"(dataset-wide observed-range proxy, no declared joint limits available)"
    )
    return CheckResult(
        check_id="episode_quality.joint_limit_saturation",
        name="Joint-limit saturation",
        severity=severity,
        summary=summary,
        citation_keys=(DQAF.key,),
        claim_type=ClaimType.RISK,
        details={
            "per_dim_saturation_frac": per_dim,
            "worst_dim": worst_dim,
            "excluded_discrete_dims": excluded_discrete,
        },
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
    # diluted into invisibility by the episode's total duration. Vectorized
    # via searchsorted (reversal_times is sorted): for window start t0 =
    # reversal_times[i], the count in [t0, t0+WINDOW) is the insertion index
    # of t0+WINDOW minus i -- avoids an O(reversals^2) Python double loop.
    if total_reversals == 0:
        peak_rate_hz = 0.0
    else:
        window_end_idx = np.searchsorted(reversal_times, reversal_times + CHATTER_WINDOW_S, side="left")
        counts = window_end_idx - np.arange(len(reversal_times))
        peak_rate_hz = counts.max() / CHATTER_WINDOW_S

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
        claim_type=ClaimType.RISK,
        details={"reversals": total_reversals, "peak_rate_hz": round(peak_rate_hz, 3)},
    )


def action_state_consistency(trajectory: Trajectory) -> CheckResult:
    """Element-wise action/state tracking error -- but only where that
    comparison is commensurable at all (GRADING_TAXONOMY_V1.md section 2C,
    the check's own origin story: it used to hard-FAIL every delta-action
    dataset, e.g. BridgeData2, because a zero-centered delta action and an
    absolute state are numerically incomparable, not because either is wrong).

    A dim-count mismatch is an architectural property, not a defect: it's
    typed CHARACTERISTIC and never reaches the element-wise comparison at all.
    A dim-count match still isn't enough on its own -- action and state can
    agree in dimensionality while meaning completely different things (a
    delta-EEF action vs. an absolute joint state). The element-wise compare
    only runs when the trajectory *declares* `action_space == "absolute"`;
    every other value (delta/velocity/torque/tokenized) and the honest default
    of undeclared (None) route to NOT_ASSESSED naming the missing declaration.
    No ingest adapter in this codebase populates `action_space` yet (see
    ingest/base.py), so today this element-wise path never runs at all --
    which is the point: it must never again FAIL a dataset whose action
    convention it doesn't actually know.
    """
    state, action = trajectory.state, trajectory.action
    if state is None or action is None:
        return _na("episode_quality.action_state_consistency", "Action-state consistency")

    if state.shape[1] != action.shape[1]:
        return CheckResult(
            check_id="episode_quality.action_state_consistency",
            name="Action-state consistency",
            severity=Severity.INFO,
            summary=(
                f"action dim ({action.shape[1]}) != state dim ({state.shape[1]}) -- action and state "
                "spaces have different dimensionality, an architectural property, not a defect"
            ),
            citation_keys=(DQAF.key, SCIZOR.key),
            claim_type=ClaimType.CHARACTERISTIC,
            details={"state_dim": int(state.shape[1]), "action_dim": int(action.shape[1])},
        )

    if trajectory.action_space != "absolute":
        declared = trajectory.action_space or "undeclared"
        return CheckResult(
            check_id="episode_quality.action_state_consistency",
            name="Action-state consistency",
            severity=Severity.NOT_APPLICABLE,
            summary=(
                f"action-space semantics is {declared!r}, not a declared 'absolute' position space -- "
                "element-wise action/state comparison is only meaningful for absolute-position actions "
                "(a delta/velocity/torque/tokenized action is not numerically comparable to state)"
            ),
            citation_keys=(DQAF.key, SCIZOR.key),
            claim_type=ClaimType.NOT_ASSESSED,
            details={"declared_action_space": trajectory.action_space},
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
        claim_type=ClaimType.RISK,
        details={"consistency_score": round(consistency, 4)},
    )


def _na(check_id: str, name: str) -> CheckResult:
    return CheckResult(
        check_id=check_id, name=name, severity=Severity.NOT_APPLICABLE,
        summary="insufficient decodable state/action data for this episode",
        citation_keys=(),
        claim_type=ClaimType.NOT_ASSESSED,
    )


def run_checks(episode: Episode, dim_stats: dict[int, DimStats] | None = None) -> list[CheckResult]:
    """`dim_stats` is the dataset-wide per-dim range/discreteness computed once
    by `dataset_dim_stats` and threaded in by grading.py; it's optional here
    only so this module's own tests can call `run_checks` without building a
    whole dataset first (an empty dict just means every dim is unrecognized,
    same as "no stats yet" -- joint_limit_saturation degrades to NOT_APPLICABLE).
    """
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
        joint_limit_saturation(t, dim_stats or {}),
        gripper_chatter(t),
        action_state_consistency(t),
    ]
