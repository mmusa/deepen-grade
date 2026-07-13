# SPDX-License-Identifier: Apache-2.0
"""Reads ROS1 `.bag` and ROS2 `.db3` bags via the pure-python `rosbags` library
(no ROS install required). Both formats share one reader because `rosbags`'
`AnyReader` auto-detects the storage backend from the path.

Same two-pass shape as the mcap reader: a raw pass for hygiene (topic names,
message types, per-message timestamps -- always works), and an opportunistic
deserialize pass for JointState/Twist/TF messages that feeds the
episode-quality and tf-tree checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deepen_grade.ingest.base import Dataset, Episode, TfEdge, TopicInfo
from deepen_grade.ingest.exceptions import DatasetReadError, MissingOptionalDependencyError
from deepen_grade.ingest.ros_common import (
    JOINT_STATE_TYPES,
    TF_MESSAGE_TYPES,
    TWIST_TYPES,
    extract_joint_state,
    extract_tf_edges,
    extract_twist,
    is_action_topic,
    series_to_trajectory,
)


def _require_rosbags():
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            feature=".bag / .db3 files", extra="ros", packages=["rosbags"]
        ) from exc
    return AnyReader


def _resolve_bag_paths(path: Path) -> list[Path]:
    """ROS1 .bag is a single file; ROS2 db3 bags are a directory containing
    metadata.yaml + one or more .db3 files. Accept either a direct .db3 file
    (walk up to its parent bag directory) or the directory itself."""
    if path.suffix == ".bag":
        return [path]
    if path.suffix == ".db3":
        candidate_dir = path.parent
        if (candidate_dir / "metadata.yaml").exists():
            return [candidate_dir]
        # No metadata.yaml sibling -- rosbags can sometimes still read the
        # bare .db3, so hand it the file directly and let it raise if not.
        return [path]
    return [path]


def read_ros_bag(path: str | Path, fmt: str) -> Dataset:
    path = Path(path)
    AnyReader = _require_rosbags()
    bag_paths = _resolve_bag_paths(path)

    topics: dict[str, dict] = {}
    warnings: list[str] = []
    state_series: list[tuple[float, np.ndarray]] = []
    action_series: list[tuple[float, np.ndarray]] = []
    gripper_series: list[tuple[float, float]] = []
    state_labels: list[str] | None = None
    tf_edges: list[TfEdge] = []

    try:
        with AnyReader(bag_paths) as reader:
            decodable_types = JOINT_STATE_TYPES | TWIST_TYPES | TF_MESSAGE_TYPES
            for connection, timestamp, rawdata in reader.messages():
                t = timestamp / 1e9
                info = topics.setdefault(
                    connection.topic, {"msg_type": connection.msgtype, "timestamps": []}
                )
                info["timestamps"].append(t)

                if connection.msgtype not in decodable_types:
                    continue
                try:
                    msg = reader.deserialize(rawdata, connection.msgtype)
                except Exception:  # noqa: BLE001 -- a single bad message shouldn't abort ingestion
                    continue

                if connection.msgtype in JOINT_STATE_TYPES:
                    js = extract_joint_state(msg)
                    if js is None:
                        continue
                    if "gripper" in connection.topic.lower():
                        gripper_series.append((t, float(np.mean(js["position"]))))
                    elif is_action_topic(connection.topic):
                        action_series.append((t, js["position"]))
                    else:
                        state_series.append((t, js["position"]))
                        if js["names"] and state_labels is None:
                            state_labels = js["names"]
                elif connection.msgtype in TWIST_TYPES and is_action_topic(connection.topic):
                    twist = extract_twist(msg)
                    if twist is not None:
                        action_series.append((t, twist))
                elif connection.msgtype in TF_MESSAGE_TYPES:
                    is_static = "static" in connection.topic.lower()
                    for parent, child, static in extract_tf_edges(msg, is_static):
                        tf_edges.append(TfEdge(parent, child, static))
    except FileNotFoundError as exc:
        raise DatasetReadError(f"{path}: {exc}") from exc

    if not topics:
        raise DatasetReadError(f"{path}: no messages found -- is this a valid {fmt} bag?")

    topic_infos = [
        TopicInfo(
            name=name,
            msg_type=info["msg_type"],
            count=len(info["timestamps"]),
            timestamps_s=np.asarray(sorted(info["timestamps"])),
        )
        for name, info in topics.items()
    ]

    if not (state_series or action_series):
        warnings.append(
            "No sensor_msgs/JointState (or Twist action) topics found: episode-quality "
            "checks skipped for this bag. This is expected for bags that only log "
            "sensors/video, not joint state."
        )
    trajectory = series_to_trajectory(state_series, state_labels, action_series, gripper_series)

    all_stamps = np.concatenate([t.timestamps_s for t in topic_infos])
    duration = float(all_stamps.max() - all_stamps.min()) if len(all_stamps) else None

    episode = Episode(
        episode_id=path.stem,
        topics=topic_infos,
        tf_edges=tf_edges or None,
        trajectory=trajectory,
        calibrations=[],
        duration_s=duration,
    )
    ds = Dataset(source=str(path), format=fmt, episodes=[episode])
    ds.warnings.extend(warnings)
    return ds


def read_ros1_bag(path: str | Path) -> Dataset:
    return read_ros_bag(path, fmt="ros1_bag")


def read_ros2_db3(path: str | Path) -> Dataset:
    return read_ros_bag(path, fmt="ros2_db3")
