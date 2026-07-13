# SPDX-License-Identifier: Apache-2.0
"""One shared, format-agnostic synthetic episode used to build the .mcap,
.bag, .db3, and LeRobot fixtures that exercise the full ingest -> checks ->
grading pipeline in the integration tests (and in the manual CLI validation
runs -- see repo README's "Development"/validation notes).

Deliberately engineered to hit every severity band at least once so the
fixture is also a good demo of what a real report looks like:
  - a genuine 2s stall (idle/stall -> WARN)
  - joint_2 pinned at its max for the last ~2s (joint-limit saturation -> WARN/FAIL)
  - a short 8Hz gripper chatter burst (gripper chatter -> FAIL)
  - the camera stream starting 25ms after joint_states (cross-modal sync -> WARN)
  - action = state + small noise (action-state consistency -> PASS)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N = 200
DT = 0.05  # 20 Hz control loop, ~10s episode
JOINT_LABELS = ["joint_1", "joint_2", "joint_3"]


@dataclass
class Scenario:
    t: np.ndarray  # (N,)
    joint_pos: np.ndarray  # (N, 3)
    action_pos: np.ndarray  # (N, 3)
    joint_labels: list[str]
    gripper_t: np.ndarray  # own clock, same rate but offset
    gripper_pos: np.ndarray  # (Ng,)
    camera_t: np.ndarray  # own clock, offset start


def build_scenario(seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    t = np.arange(N) * DT

    j1 = np.sin(0.5 * t)
    j3 = 0.5 * np.cos(0.3 * t)
    j2 = np.empty(N)
    j2[:120] = np.linspace(-1.0, 0.5, 120)
    j2[120:160] = j2[119]  # part of the stall window, see below
    j2[160:] = np.linspace(0.5, 1.0, 40)
    j2[180:] = 1.0  # pinned at max for the last ~1s -> sustained saturation

    joint_pos = np.stack([j1, j2, j3], axis=1)
    # Stall: freeze ALL joints for idx 120..159 (2s of a ~10s episode = 20%).
    joint_pos[120:160] = joint_pos[119]

    action_pos = joint_pos + rng.normal(scale=0.01, size=joint_pos.shape)

    gripper_t = t.copy()
    gripper_pos = np.full(N, 0.0)
    gripper_pos[:40] = np.linspace(0.0, 1.0, 40)  # one smooth close
    chatter_slice = slice(40, 70)
    gripper_pos[chatter_slice] = 0.5 + 0.5 * np.sign(
        np.sin(2 * np.pi * 8.0 * (t[chatter_slice] - t[40]))
    )  # ~8Hz chatter burst
    gripper_pos[70:] = 1.0

    camera_t = t + 0.025  # camera driver comes up 25ms "late" -> WARN-band skew

    return Scenario(
        t=t, joint_pos=joint_pos, action_pos=action_pos, joint_labels=list(JOINT_LABELS),
        gripper_t=gripper_t, gripper_pos=gripper_pos, camera_t=camera_t,
    )
