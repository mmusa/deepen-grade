# SPDX-License-Identifier: Apache-2.0
"""Format-agnostic data model that every ingestion adapter normalizes into.

Every check in `deepen_grade.checks` operates on these dataclasses, never on a
raw `.mcap`/`.bag`/`.db3` file or a LeRobot parquet frame directly. This is the
port-and-adapter boundary: adding a new source format means writing a new
adapter under `ingest/` that fills in a `Dataset`; it never touches a check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Keyword buckets used to classify a raw topic/feature name into a sensing
# modality for cross-modal timestamp-skew analysis. Order matters: first match
# wins. This is a coarse, transparent heuristic (documented in the README), not
# a learned or proprietary classifier.
_MODALITY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lidar", ("lidar", "points", "pointcloud", "scan", "velodyne", "ouster")),
    ("vision", ("image", "camera", "cam", "rgb", "depth", "video")),
    ("proprioception", ("joint", "state", "odom", "imu", "gripper", "wrench", "force")),
    ("control", ("cmd", "command", "action", "target")),
    ("tf", ("tf",)),
)


def classify_modality(topic_or_feature_name: str) -> str:
    """Best-effort modality classification from a topic/feature name.

    Returns one of: lidar, vision, proprioception, control, tf, other.
    """
    name = topic_or_feature_name.lower()
    for modality, keywords in _MODALITY_KEYWORDS:
        if any(kw in name for kw in keywords):
            return modality
    return "other"


@dataclass
class TopicInfo:
    """One data stream (a ROS topic, or a LeRobot feature) within an episode."""

    name: str
    msg_type: str | None
    count: int
    timestamps_s: np.ndarray  # (N,) seconds, monotonically non-decreasing
    modality: str = field(default="")

    def __post_init__(self) -> None:
        if not self.modality:
            self.modality = classify_modality(self.name)


@dataclass
class TfEdge:
    parent_frame: str
    child_frame: str
    is_static: bool


@dataclass
class Trajectory:
    """Normalized per-episode kinematic data used by episode-quality checks.

    All arrays share the same time axis (`timestamps_s`) once resampled by the
    ingestion adapter onto the observation rate. `state_*` reflects the recorded
    proprioceptive state; `action` reflects the commanded action (may be absent).
    """

    timestamps_s: np.ndarray  # (T,)
    state: np.ndarray | None  # (T, D) generic proprioceptive state (joints/pose/etc.)
    state_labels: list[str] | None
    action: np.ndarray | None  # (T, A) commanded action, same convention as state where possible
    action_labels: list[str] | None
    gripper_position: np.ndarray | None  # (T,) or (T, G)


@dataclass
class CalibrationInfo:
    camera_id: str
    intrinsics: dict | None  # e.g. {"fx","fy","cx","cy","width","height"}
    extrinsics: list[float] | None  # 6-vector [x,y,z,rx,ry,rz] or flattened 4x4
    source: str  # where this was found, e.g. "metadata.json:ext1_cam_extrinsics"


@dataclass
class Episode:
    episode_id: str
    topics: list[TopicInfo] = field(default_factory=list)
    tf_edges: list[TfEdge] | None = None  # None => format has no tf concept (e.g. LeRobot)
    trajectory: Trajectory | None = None
    calibrations: list[CalibrationInfo] = field(default_factory=list)
    duration_s: float | None = None
    success: bool | None = None
    task: str | None = None


@dataclass
class Dataset:
    source: str
    format: str  # "mcap" | "ros1_bag" | "ros2_db3" | "lerobot"
    episodes: list[Episode]
    warnings: list[str] = field(default_factory=list)
