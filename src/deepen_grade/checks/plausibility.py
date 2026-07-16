# SPDX-License-Identifier: Apache-2.0
"""Dataset-level plausibility check: does this even look like robot-learning data?

Every other check in this package asks "is the *structure* sound?" (right
columns, right dtypes, consistent dims). None of them ask "is the *content*
plausibly a robot trajectory?" -- a dataset can parse as a perfectly valid
LeRobot/mcap/bag file while containing something else entirely (a medical
imaging export, a non-robotics tabular dump reshaped to fit the schema). This
check is the one cheap, format-agnostic guard against that: no state/action
signal anywhere, a state that never actually moves, or no usable timestamps.

Severity is always INFO (or N/A when nothing looks wrong) and is deliberately
uncited -- see the exemption in checks/base.py -- because "does this look like
robot data" isn't a metric any of the cited papers define; it never affects
the score either way (INFO carries a zero-point penalty in grading.py).
"""

from __future__ import annotations

import numpy as np

from deepen_grade.checks.base import CheckResult, Severity
from deepen_grade.ingest.base import Dataset

CHECK_ID = "plausibility.robot_data"

# A state dim whose value never moves by more than this across an entire
# episode is "constant" for plausibility purposes -- real joint/pose
# trajectories move; content that merely happens to fit the schema (e.g. a
# per-row label reused as a fake "state") typically doesn't.
CONSTANT_DIM_TOLERANCE = 1e-9


def _all_dims_constant(state: np.ndarray) -> bool:
    return bool(np.all(np.ptp(state, axis=0) <= CONSTANT_DIM_TOLERANCE))


def robot_data_plausibility(dataset: Dataset) -> CheckResult:
    with_trajectory = [ep for ep in dataset.episodes if ep.trajectory is not None]

    if not with_trajectory:
        reason = "no state/action trajectories found in any episode"
    else:
        no_signal = all(
            ep.trajectory.state is None and ep.trajectory.action is None for ep in with_trajectory
        )
        no_timestamps = all(len(ep.trajectory.timestamps_s) < 2 for ep in with_trajectory)
        state_episodes = [ep for ep in with_trajectory if ep.trajectory.state is not None]
        all_constant = bool(state_episodes) and all(
            _all_dims_constant(ep.trajectory.state) for ep in state_episodes
        )
        reason = (
            "no state/action trajectories found in any episode" if no_signal else
            "state values never change across any sampled episode" if all_constant else
            "no usable timestamps in any episode" if no_timestamps else
            None
        )

    if reason is None:
        return CheckResult(
            check_id=CHECK_ID,
            name="Robot-data plausibility",
            severity=Severity.NOT_APPLICABLE,
            summary="content plausibly looks like robot-learning data",
            citation_keys=(),
        )

    return CheckResult(
        check_id=CHECK_ID,
        name="Robot-data plausibility",
        severity=Severity.INFO,
        summary=(
            f"structure parses as {dataset.format} but content does not look like "
            f"robot-learning data ({reason})"
        ),
        citation_keys=(),  # deliberately uncited heuristic -- see module docstring
        details={"reason": reason, "episodes_checked": len(with_trajectory)},
    )


def run_checks(dataset: Dataset) -> list[CheckResult]:
    return [robot_data_plausibility(dataset)]
