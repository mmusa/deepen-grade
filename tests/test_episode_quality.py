# SPDX-License-Identifier: Apache-2.0
import numpy as np

from deepen_grade.checks import episode_quality as eq
from deepen_grade.checks.base import ClaimType, Severity
from deepen_grade.ingest.base import Episode, Trajectory


def _traj(state=None, action=None, gripper=None, n=200, dt=0.02, action_space=None):
    t = np.arange(n) * dt
    return Trajectory(
        timestamps_s=t, state=state, state_labels=None, action=action,
        action_labels=None, gripper_position=gripper, action_space=action_space,
    )


def _dim_stats(*trajectories: Trajectory) -> dict[int, eq.DimStats]:
    """Dataset-wide dim stats over the given trajectories -- what grading.py
    would compute for a dataset containing exactly these episodes."""
    episodes = [Episode(episode_id=f"e{i}", trajectory=t) for i, t in enumerate(trajectories)]
    return eq.dataset_dim_stats(episodes)


def test_idle_stall_na_without_trajectory():
    result = eq.idle_stall(_traj(state=None, action=None))
    assert result.severity == Severity.NOT_APPLICABLE


def test_idle_stall_detects_mostly_idle_episode():
    n = 200
    pos = np.zeros((n, 3))
    pos[150:, 0] = np.linspace(0, 1, n - 150)  # only moves in the last 25%
    result = eq.idle_stall(_traj(state=pos, n=n))
    assert result.details["stall_frac"] > 0.5
    assert result.severity == Severity.FAIL
    # Idle/stall is a training-value proxy (SCIZOR curates on it), never a
    # correctness claim -- it must never be able to move the letter grade.
    assert result.claim_type == ClaimType.RISK


def test_idle_stall_passes_for_continuous_motion():
    n = 200
    t = np.arange(n) * 0.02
    pos = np.stack([np.sin(t), np.cos(t), t], axis=1)
    result = eq.idle_stall(_traj(state=pos, n=n))
    assert result.severity == Severity.PASS


def test_jerk_smoothness_smooth_vs_jerky():
    n = 300
    t = np.arange(n) * 0.02
    smooth = np.stack([np.sin(0.5 * t)] * 3, axis=1)
    rng = np.random.default_rng(0)
    jerky = np.cumsum(rng.normal(scale=1.0, size=(n, 3)), axis=0)

    smooth_result = eq._log_dimensionless_jerk(smooth, 0.02)
    jerky_result = eq._log_dimensionless_jerk(jerky, 0.02)
    assert smooth_result is not None and jerky_result is not None
    assert smooth_result > jerky_result  # smoother motion -> higher (less negative) LDJ


def test_joint_limit_saturation_detects_sustained_edge_dwell():
    n = 200
    pos = np.zeros((n, 2))
    pos[:, 0] = np.linspace(-1, 1, n)
    # dim 1 wiggles across a real range (many unique values, so it isn't
    # excluded as discrete) but is pinned at its max for the back half.
    pos[:100, 1] = np.linspace(-1, 0.9, 100)
    pos[100:, 1] = 1.0
    traj = _traj(state=pos, n=n)
    result = eq.joint_limit_saturation(traj, _dim_stats(traj))
    assert result.severity in (Severity.WARN, Severity.FAIL)
    assert result.details["worst_dim"] == "dim_1"
    # A proxy against an observed (never declared) range -- always RISK, per
    # GRADING_TAXONOMY_V1.md section 2C: "no declared joint limits => RISK at
    # most, never DEFECT."
    assert result.claim_type == ClaimType.RISK


def test_joint_limit_saturation_passes_smooth_sweep():
    n = 200
    pos = np.linspace(-1, 1, n).reshape(-1, 1)
    traj = _traj(state=pos, n=n)
    result = eq.joint_limit_saturation(traj, _dim_stats(traj))
    assert result.severity == Severity.PASS


def test_joint_limit_saturation_excludes_binary_dim():
    """The bug this fix closes: a binary 0/1 gripper flag has full excursion
    (0 to 1) but only 2 distinct values, so every sample sits "at the limit"
    by construction -- not a real finding. It must be excluded outright, never
    reported as 100% saturated."""
    n = 200
    pos = np.zeros((n, 2))
    pos[:, 0] = np.sin(np.linspace(0, 4 * np.pi, n))  # smooth, unsaturated joint
    pos[:, 1] = np.where(np.arange(n) % 2 == 0, 0.0, 1.0)  # binary gripper flag
    traj = _traj(state=pos, n=n)
    result = eq.joint_limit_saturation(traj, _dim_stats(traj))
    assert "dim_1" not in result.details["per_dim_saturation_frac"]
    assert "dim_1" in result.details["excluded_discrete_dims"]
    assert result.severity == Severity.PASS


def test_joint_limit_saturation_uses_dataset_wide_range_not_episode_own():
    """The other bug this fix closes: a short, low-variance episode that only
    ever visits a narrow slice of the joint's true range must not read as
    saturated just because that slice is close to *its own* local min/max --
    only the dataset-wide range matters."""
    n = 200
    wide_range = np.linspace(-1, 1, n).reshape(-1, 1)  # establishes the true [-1, 1] range
    narrow_slice = (0.5 + 0.01 * np.sin(np.linspace(0, 20 * np.pi, n))).reshape(-1, 1)  # hugs 0.5, nowhere near +/-1

    wide_traj = _traj(state=wide_range, n=n)
    narrow_traj = _traj(state=narrow_slice, n=n)
    dim_stats = _dim_stats(wide_traj, narrow_traj)

    result = eq.joint_limit_saturation(narrow_traj, dim_stats)
    assert result.severity == Severity.PASS


def test_gripper_chatter_detects_oscillation():
    n = 200
    t = np.arange(n) * 0.02
    chattering = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 8 * t))  # 8Hz square wave
    result = eq.gripper_chatter(_traj(gripper=chattering, n=n))
    assert result.severity == Severity.FAIL
    assert result.details["peak_rate_hz"] > eq.CHATTER_FAIL_HZ
    assert result.claim_type == ClaimType.RISK


def test_gripper_chatter_passes_single_close():
    n = 200
    t = np.arange(n) * 0.02
    grip = np.clip(t / t.max(), 0, 1)  # one smooth close
    result = eq.gripper_chatter(_traj(gripper=grip, n=n))
    assert result.severity == Severity.PASS


def test_action_state_consistency_matches():
    n = 100
    state = np.random.default_rng(1).normal(size=(n, 4))
    traj = _traj(state=state, action=state.copy(), n=n, action_space="absolute")
    result = eq.action_state_consistency(traj)
    assert result.severity == Severity.PASS
    assert result.details["consistency_score"] > 0.99
    assert result.claim_type == ClaimType.RISK


def test_action_state_consistency_dim_mismatch_is_characteristic_not_defect():
    """The bug this fix closes: BridgeData2-style zero-centered delta actions
    (or any action/state space of different width) used to hard-FAIL every
    such episode. Differing dimensionality is architectural, not a defect."""
    n = 50
    state = np.zeros((n, 4))
    action = np.zeros((n, 6))
    result = eq.action_state_consistency(_traj(state=state, action=action, n=n))
    assert result.claim_type == ClaimType.CHARACTERISTIC
    assert result.severity != Severity.FAIL
    assert "architectural property" in result.summary


def test_action_state_consistency_not_assessed_when_action_space_undeclared():
    """The other half of the fix: even with matching dims, element-wise
    comparison must not run against an undeclared (None) action space -- e.g.
    a delta-EEF action can equal state numerically by coincidence and still
    mean something completely different."""
    n = 100
    state = np.random.default_rng(1).normal(size=(n, 3))
    result = eq.action_state_consistency(_traj(state=state, action=state.copy(), n=n, action_space=None))
    assert result.severity == Severity.NOT_APPLICABLE
    assert result.claim_type == ClaimType.NOT_ASSESSED
    assert "absolute" in result.summary


def test_action_state_consistency_not_assessed_for_delta_action_space():
    n = 100
    rng = np.random.default_rng(2)
    state = rng.normal(size=(n, 3))
    action = rng.normal(size=(n, 3)) * 10  # would have diverged under the old unconditional compare
    result = eq.action_state_consistency(_traj(state=state, action=action, n=n, action_space="delta"))
    assert result.severity == Severity.NOT_APPLICABLE
    assert result.claim_type == ClaimType.NOT_ASSESSED
    assert result.details["declared_action_space"] == "delta"


def test_action_state_consistency_diverges_when_declared_absolute():
    n = 100
    rng = np.random.default_rng(2)
    state = rng.normal(size=(n, 3))
    action = rng.normal(size=(n, 3)) * 10  # unrelated, much larger scale
    traj = _traj(state=state, action=action, n=n, action_space="absolute")
    result = eq.action_state_consistency(traj)
    assert result.severity in (Severity.WARN, Severity.FAIL)
    assert result.claim_type == ClaimType.RISK


def test_run_checks_all_na_without_trajectory():
    ep = Episode(episode_id="e", trajectory=None)
    results = eq.run_checks(ep)
    assert len(results) == 5
    assert all(r.severity == Severity.NOT_APPLICABLE for r in results)
    assert all(r.claim_type == ClaimType.NOT_ASSESSED for r in results)
