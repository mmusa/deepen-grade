# SPDX-License-Identifier: Apache-2.0
import numpy as np

from deepen_grade.checks import episode_quality as eq
from deepen_grade.checks.base import Severity
from deepen_grade.ingest.base import Trajectory


def _traj(state=None, action=None, gripper=None, n=200, dt=0.02):
    t = np.arange(n) * dt
    return Trajectory(
        timestamps_s=t, state=state, state_labels=None, action=action,
        action_labels=None, gripper_position=gripper,
    )


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
    pos[100:, 1] = 1.0  # dim 1 pinned at its max for the back half
    result = eq.joint_limit_saturation(_traj(state=pos, n=n))
    assert result.severity in (Severity.WARN, Severity.FAIL)
    assert result.details["worst_dim"] == "dim_1"


def test_joint_limit_saturation_passes_smooth_sweep():
    n = 200
    pos = np.linspace(-1, 1, n).reshape(-1, 1)
    result = eq.joint_limit_saturation(_traj(state=pos, n=n))
    assert result.severity == Severity.PASS


def test_gripper_chatter_detects_oscillation():
    n = 200
    t = np.arange(n) * 0.02
    chattering = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 8 * t))  # 8Hz square wave
    result = eq.gripper_chatter(_traj(gripper=chattering, n=n))
    assert result.severity == Severity.FAIL
    assert result.details["peak_rate_hz"] > eq.CHATTER_FAIL_HZ


def test_gripper_chatter_passes_single_close():
    n = 200
    t = np.arange(n) * 0.02
    grip = np.clip(t / t.max(), 0, 1)  # one smooth close
    result = eq.gripper_chatter(_traj(gripper=grip, n=n))
    assert result.severity == Severity.PASS


def test_action_state_consistency_matches():
    n = 100
    state = np.random.default_rng(1).normal(size=(n, 4))
    result = eq.action_state_consistency(_traj(state=state, action=state.copy(), n=n))
    assert result.severity == Severity.PASS
    assert result.details["consistency_score"] > 0.99


def test_action_state_consistency_dim_mismatch():
    n = 50
    state = np.zeros((n, 4))
    action = np.zeros((n, 6))
    result = eq.action_state_consistency(_traj(state=state, action=action, n=n))
    assert result.severity == Severity.FAIL
    assert "action dim" in result.summary


def test_action_state_consistency_diverges():
    n = 100
    rng = np.random.default_rng(2)
    state = rng.normal(size=(n, 3))
    action = rng.normal(size=(n, 3)) * 10  # unrelated, much larger scale
    result = eq.action_state_consistency(_traj(state=state, action=action, n=n))
    assert result.severity in (Severity.WARN, Severity.FAIL)


def test_run_checks_all_na_without_trajectory():
    from deepen_grade.ingest.base import Episode

    ep = Episode(episode_id="e", trajectory=None)
    results = eq.run_checks(ep)
    assert len(results) == 5
    assert all(r.severity == Severity.NOT_APPLICABLE for r in results)
