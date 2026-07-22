# SPDX-License-Identifier: Apache-2.0
import numpy as np

from deepen_grade.checks import plausibility as pl
from deepen_grade.checks.base import ClaimType, Severity
from deepen_grade.ingest.base import Dataset, Episode, Trajectory


def _episode(episode_id, state=None, action=None, n=50):
    t = np.arange(n) * 0.02
    traj = None
    if state is not None or action is not None:
        traj = Trajectory(timestamps_s=t, state=state, state_labels=None, action=action,
                           action_labels=None, gripper_position=None)
    return Episode(episode_id=episode_id, trajectory=traj)


def _moving_state(n=50, seed=0):
    t = np.arange(n) * 0.02
    return np.stack([np.sin(t), np.cos(t)], axis=1)


def test_fires_when_no_trajectory_anywhere():
    ds = Dataset(source="x", format="lerobot", episodes=[_episode("a"), _episode("b")])
    result = pl.robot_data_plausibility(ds)
    assert result.severity == Severity.INFO
    assert "no state/action trajectories" in result.summary
    assert result.citation_keys == ()  # deliberately uncited
    assert result.claim_type == ClaimType.CHARACTERISTIC


def test_fires_when_state_never_moves():
    constant = np.zeros((50, 3))
    ds = Dataset(source="x", format="lerobot", episodes=[_episode("a", state=constant)])
    result = pl.robot_data_plausibility(ds)
    assert result.severity == Severity.INFO
    assert "never change" in result.summary


def test_fires_when_no_usable_timestamps():
    ep = _episode("a", state=_moving_state(n=1), n=1)
    ds = Dataset(source="x", format="lerobot", episodes=[ep])
    result = pl.robot_data_plausibility(ds)
    assert result.severity == Severity.INFO


def test_does_not_fire_on_plausible_robot_data():
    ds = Dataset(source="x", format="lerobot", episodes=[_episode("a", state=_moving_state())])
    result = pl.robot_data_plausibility(ds)
    assert result.severity == Severity.NOT_APPLICABLE
    assert result.citation_keys == ()


def test_does_not_fire_on_standard_fixture_dataset(sample_lerobot):
    from deepen_grade.ingest.detect import load_dataset

    dataset = load_dataset(str(sample_lerobot))
    result = pl.robot_data_plausibility(dataset)
    assert result.severity == Severity.NOT_APPLICABLE


def test_wired_into_dataset_level_grading_and_never_affects_score():
    from deepen_grade.grading import grade_dataset, score_penalty

    ds = Dataset(source="x", format="lerobot", episodes=[_episode("a", state=np.zeros((50, 2)))])
    grade = grade_dataset(ds)
    plausibility_result = next(r for r in grade.dataset_level_results if r.check_id == pl.CHECK_ID)
    assert plausibility_result.severity == Severity.INFO

    # INFO carries a zero-point penalty (grading.py) -- firing must not move the score:
    # dropping it from the dataset-level results changes the penalty by exactly zero.
    with_it = score_penalty(grade.dataset_level_results)
    without_it = score_penalty([r for r in grade.dataset_level_results if r.check_id != pl.CHECK_ID])
    assert with_it == without_it
