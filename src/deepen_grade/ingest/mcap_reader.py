# SPDX-License-Identifier: Apache-2.0
"""Reads `.mcap` files -- the primary, ROS 2-default ingestion format.

Two independent passes over the file:

1. A raw pass (`mcap.reader.make_reader`, no decoding) that builds per-topic
   timestamp series and schema names. This always works on any valid mcap
   file and is all the hygiene checks need.
2. An opportunistic decode pass (`mcap-ros2-support`, when installed and the
   file's channels use `ros2msg` schema encoding) that pulls JointState /
   Twist / TF messages out for the episode-quality and tf-tree checks. If
   decoding isn't available or the file has no such topics, this pass is
   skipped and hygiene-only results are returned -- never a hard failure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deepen_grade.ingest.base import CalibrationInfo, Dataset, Episode, TfEdge, TopicInfo, Trajectory
from deepen_grade.ingest.exceptions import DatasetReadError, MissingOptionalDependencyError
from deepen_grade.ingest.ros_common import (
    CAMERA_INFO_TYPES,
    JOINT_STATE_TYPES,
    TF_MESSAGE_TYPES,
    TWIST_TYPES,
    build_calibrations,
    extract_camera_info,
    extract_joint_state,
    extract_static_transforms,
    extract_tf_edges,
    extract_twist,
    frame_id_of,
    is_action_topic,
    series_to_trajectory,
)


def _require_mcap():
    try:
        from mcap.reader import make_reader
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            feature=".mcap files", extra="mcap", packages=["mcap"]
        ) from exc
    return make_reader


def read_mcap(path: str | Path) -> Dataset:
    path = Path(path)
    make_reader = _require_mcap()

    topics: dict[str, dict] = {}

    with open(path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            info = topics.setdefault(
                channel.topic,
                {"msg_type": schema.name if schema else None, "timestamps": []},
            )
            info["timestamps"].append(message.log_time / 1e9)

    if not topics:
        raise DatasetReadError(f"{path}: no messages found -- is this a valid mcap file?")

    topic_infos = [
        TopicInfo(
            name=name,
            msg_type=info["msg_type"],
            count=len(info["timestamps"]),
            timestamps_s=np.asarray(sorted(info["timestamps"])),
        )
        for name, info in topics.items()
    ]

    trajectory, tf_edges, calibrations, warnings = _decode_semantics(path, topics.keys())

    all_stamps = np.concatenate([t.timestamps_s for t in topic_infos])
    duration = float(all_stamps.max() - all_stamps.min()) if len(all_stamps) else None

    episode = Episode(
        episode_id=path.stem,
        topics=topic_infos,
        tf_edges=tf_edges,
        trajectory=trajectory,
        calibrations=calibrations,
        duration_s=duration,
    )
    ds = Dataset(source=str(path), format="mcap", episodes=[episode])
    ds.warnings.extend(warnings)
    return ds


def _decode_semantics(
    path: Path, topic_names
) -> tuple[Trajectory | None, list[TfEdge] | None, list[CalibrationInfo], list[str]]:
    """Best-effort decode of JointState/Twist/TF/CameraInfo from ros2msg-encoded channels.

    Returns (trajectory_or_None, tf_edges_or_None, calibrations, warnings). Never
    raises -- any failure here degrades to "episode-quality checks skipped", not
    a crash.
    """
    warnings: list[str] = []
    try:
        from mcap_ros2.decoder import DecoderFactory
    except ImportError:
        warnings.append(
            "mcap-ros2-support not installed: episode-quality and tf-tree checks "
            "skipped for this mcap file (hygiene checks still ran). "
            'Install with `pip install "deepen-grade[mcap]"`.'
        )
        return None, None, [], warnings

    state_series: list[tuple[float, np.ndarray]] = []
    action_series: list[tuple[float, np.ndarray]] = []
    gripper_series: list[tuple[float, float]] = []
    state_labels: list[str] | None = None
    tf_edges: list[TfEdge] = []
    static_transforms: dict[str, list[float]] = {}
    camera_intrinsics: dict[str, tuple[str | None, dict]] = {}  # topic -> (frame_id, intrinsics)
    saw_decodable = False

    try:
        from mcap.reader import make_reader

        with open(path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                if ros_msg is None or schema is None:
                    continue
                t = message.log_time / 1e9
                if schema.name in JOINT_STATE_TYPES:
                    js = extract_joint_state(ros_msg)
                    if js is None:
                        continue
                    saw_decodable = True
                    if "gripper" in channel.topic.lower():
                        gripper_series.append((t, float(np.mean(js["position"]))))
                    elif is_action_topic(channel.topic):
                        action_series.append((t, js["position"]))
                    else:
                        state_series.append((t, js["position"]))
                        if state_labels is None and js["names"]:
                            state_labels = js["names"]
                elif schema.name in TWIST_TYPES and is_action_topic(channel.topic):
                    twist = extract_twist(ros_msg)
                    if twist is not None:
                        saw_decodable = True
                        action_series.append((t, twist))
                elif schema.name in TF_MESSAGE_TYPES:
                    is_static = "static" in channel.topic.lower()
                    for parent, child, static in extract_tf_edges(ros_msg, is_static):
                        tf_edges.append(TfEdge(parent, child, static))
                    if is_static:
                        static_transforms.update(extract_static_transforms(ros_msg))
                elif schema.name in CAMERA_INFO_TYPES and channel.topic not in camera_intrinsics:
                    intrinsics = extract_camera_info(ros_msg)  # first message per topic is enough
                    if intrinsics is not None:
                        camera_intrinsics[channel.topic] = (frame_id_of(ros_msg), intrinsics)
    except Exception as exc:  # noqa: BLE001 -- decode is opportunistic, never fatal
        warnings.append(f"mcap decode pass failed ({exc.__class__.__name__}: {exc}); "
                         "falling back to hygiene-only results.")
        return None, (tf_edges or None), [], warnings

    calibrations = build_calibrations(camera_intrinsics, static_transforms)

    if not saw_decodable:
        warnings.append(
            "No sensor_msgs/JointState (or Twist action) topics decodable in this mcap "
            "file: episode-quality checks skipped. This is expected for non-ROS2msg "
            "encodings (e.g. protobuf/json) or datasets that don't publish JointState."
        )
        return None, (tf_edges or None), calibrations, warnings

    trajectory = series_to_trajectory(state_series, state_labels, action_series, gripper_series)
    return trajectory, (tf_edges or None), calibrations, warnings
