# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for pulling generic robotics semantics out of decoded ROS
messages, used by both the `.mcap` reader and the `.bag`/`.db3` reader so the
two don't duplicate this logic.

Decoded message objects from both `mcap-ros2-support` and `rosbags` expose
plain attribute access (`msg.position`, `msg.header.stamp`, ...), so these
helpers work against either without caring which library produced the object.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from deepen_grade.ingest.base import CalibrationInfo, Trajectory

# Message types (short name, ROS1 and ROS2 spellings) that carry joint-level
# proprioceptive state or commanded action. Topic *name* heuristics further
# narrow "state" vs "action" -- see classify_modality / ACTION_NAME_HINTS.
JOINT_STATE_TYPES = {
    "sensor_msgs/JointState",
    "sensor_msgs/msg/JointState",
}
TWIST_TYPES = {
    "geometry_msgs/Twist",
    "geometry_msgs/msg/Twist",
    "geometry_msgs/TwistStamped",
    "geometry_msgs/msg/TwistStamped",
}
TF_MESSAGE_TYPES = {
    "tf2_msgs/TFMessage",
    "tf2_msgs/msg/TFMessage",
    "tf/tfMessage",
}
CAMERA_INFO_TYPES = {
    "sensor_msgs/CameraInfo",
    "sensor_msgs/msg/CameraInfo",
}

ACTION_NAME_HINTS = ("cmd", "command", "target", "setpoint", "desired")


def is_action_topic(topic_name: str) -> bool:
    lowered = topic_name.lower()
    return any(hint in lowered for hint in ACTION_NAME_HINTS)


def stamp_to_seconds(stamp: Any) -> float | None:
    """Convert a ROS1 or ROS2 header stamp to float seconds, or None."""
    if stamp is None:
        return None
    # ROS2: sec/nanosec ; ROS1: secs/nsecs
    sec = getattr(stamp, "sec", None)
    nsec = getattr(stamp, "nanosec", None)
    if sec is None:
        sec = getattr(stamp, "secs", None)
        nsec = getattr(stamp, "nsecs", None)
    if sec is None:
        return None
    return float(sec) + float(nsec or 0) * 1e-9


def _as_float_array(value: Any) -> np.ndarray:
    """Array fields differ by decoder: mcap-ros2-support yields python lists,
    rosbags yields numpy arrays directly -- normalize without ever using
    `value or default` (ambiguous/truthiness-raising for numpy arrays)."""
    if value is None:
        return np.asarray([], dtype=float)
    return np.asarray(value, dtype=float)


def extract_joint_state(msg: Any) -> dict | None:
    """Pull {names, position, velocity, effort} out of a decoded JointState."""
    position = getattr(msg, "position", None)
    if position is None:
        return None
    names_raw = getattr(msg, "name", None)
    names = list(names_raw) if names_raw is not None else []
    return {
        "names": names,
        "position": _as_float_array(position),
        "velocity": _as_float_array(getattr(msg, "velocity", None)),
        "effort": _as_float_array(getattr(msg, "effort", None)),
    }


def extract_twist(msg: Any) -> np.ndarray | None:
    """Flatten a Twist(Stamped) into a 6-vector [vx,vy,vz,wx,wy,wz]."""
    twist = getattr(msg, "twist", msg)  # TwistStamped wraps a Twist
    linear = getattr(twist, "linear", None)
    angular = getattr(twist, "angular", None)
    if linear is None or angular is None:
        return None
    return np.array(
        [linear.x, linear.y, linear.z, angular.x, angular.y, angular.z], dtype=float
    )


def frame_id_of(msg: Any) -> str | None:
    """The `header.frame_id` of a decoded message, if it carries one."""
    header = getattr(msg, "header", None)
    frame_id = getattr(header, "frame_id", None) if header is not None else None
    return str(frame_id) if frame_id else None


def extract_camera_info(msg: Any) -> dict | None:
    """Pull {fx,fy,cx,cy,width,height} out of a decoded CameraInfo.

    `k` is the row-major 3x3 intrinsic matrix (ROS field name is lowercase in
    ROS2, uppercase `K` in ROS1 message text but both decoders normalize
    attribute access to the field name as written in the schema -- try both):
    fx=k[0], fy=k[4], cx=k[2], cy=k[5].
    """
    k = getattr(msg, "k", None)
    if k is None:
        k = getattr(msg, "K", None)
    if k is None:
        return None
    k = np.asarray(k, dtype=float)
    if k.size != 9:
        return None
    width, height = getattr(msg, "width", None), getattr(msg, "height", None)
    return {
        "fx": float(k[0]), "fy": float(k[4]), "cx": float(k[2]), "cy": float(k[5]),
        "width": int(width) if width else None,
        "height": int(height) if height else None,
    }


def extract_static_transforms(msg: Any) -> dict[str, list[float]]:
    """Map child_frame_id -> [tx,ty,tz,qx,qy,qz,qw] for every transform in a
    (presumed /tf_static) TFMessage.

    Used to pair a camera's extrinsics with its frame_id when CameraInfo
    itself carries no extrinsics slot (it never does -- that's normal ROS).
    Never invents a transform that wasn't actually published; a camera frame
    absent here just means extrinsics stays None (see calibration_sanity.py:
    "do NOT invent identity").
    """
    result: dict[str, list[float]] = {}
    for transform in getattr(msg, "transforms", None) or []:
        child = getattr(transform, "child_frame_id", None)
        tf = getattr(transform, "transform", None)
        if not child or tf is None:
            continue
        translation, rotation = getattr(tf, "translation", None), getattr(tf, "rotation", None)
        if translation is None or rotation is None:
            continue
        result[str(child)] = [
            float(translation.x), float(translation.y), float(translation.z),
            float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w),
        ]
    return result


def build_calibrations(
    camera_intrinsics: dict[str, tuple[str | None, dict]], static_transforms: dict[str, list[float]]
) -> list[CalibrationInfo]:
    """Pair each camera's intrinsics (topic -> (frame_id, intrinsics)) with its
    /tf_static extrinsics by frame_id. Shared by the mcap and ROS1/ROS2 bag
    readers so the pairing logic lives in exactly one place.

    A camera whose frame_id has no matching /tf_static entry still gets a
    CalibrationInfo with extrinsics=None -- calibration_sanity already reports
    that correctly as "extrinsics missing", never a fabricated identity.
    """
    calibrations = []
    for topic, (frame_id, intrinsics) in camera_intrinsics.items():
        camera_id = frame_id or topic
        extrinsics = static_transforms.get(frame_id) if frame_id else None
        calibrations.append(
            CalibrationInfo(camera_id=camera_id, intrinsics=intrinsics, extrinsics=extrinsics,
                             source="camera_info")
        )
    return calibrations


def extract_tf_edges(msg: Any, is_static: bool) -> list[tuple[str, str, bool]]:
    """Pull (parent_frame, child_frame, is_static) tuples out of a TFMessage."""
    edges = []
    transforms = getattr(msg, "transforms", None)
    for transform in (transforms if transforms is not None else []):
        header = getattr(transform, "header", None)
        parent = getattr(header, "frame_id", None) if header is not None else None
        child = getattr(transform, "child_frame_id", None)
        if parent and child:
            edges.append((str(parent), str(child), is_static))
    return edges


def _resample_onto(base_t: np.ndarray, series_t: np.ndarray, series_v: np.ndarray) -> np.ndarray:
    """Linearly resample a (possibly differently-timed) series onto base_t, per column."""
    if len(series_t) < 2:
        return np.full((len(base_t), series_v.shape[1]), np.nan)
    order = np.argsort(series_t)
    series_t, series_v = series_t[order], series_v[order]
    cols = [np.interp(base_t, series_t, series_v[:, j]) for j in range(series_v.shape[1])]
    return np.stack(cols, axis=1)


def series_to_trajectory(
    state_series: list[tuple[float, np.ndarray]],
    state_labels: list[str] | None,
    action_series: list[tuple[float, np.ndarray]],
    gripper_series: list[tuple[float, float]],
) -> Trajectory | None:
    """Build a Trajectory on a single shared clock, resampling action/gripper onto it.

    The proprioceptive *state* topic (usually the highest-rate, most complete
    stream) sets the clock; other streams are linearly interpolated onto it.
    This is the standard cross-topic alignment approach for asynchronous ROS
    data and keeps every check downstream working on equal-length arrays.
    Used by both the mcap and the ROS1/ROS2 bag readers.
    """
    if not state_series and not action_series:
        return None

    if state_series:
        timestamps = np.asarray([t for t, _ in state_series])
        state = np.asarray([v for _, v in state_series])
    else:
        timestamps = np.asarray([t for t, _ in action_series])
        state = None

    action = None
    if action_series:
        action_t = np.asarray([t for t, _ in action_series])
        action_v = np.asarray([v for _, v in action_series])
        action = action_v if state is None else _resample_onto(timestamps, action_t, action_v)

    gripper = None
    if gripper_series:
        grip_t = np.asarray([t for t, _ in gripper_series])
        grip_v = np.asarray([[v] for _, v in gripper_series])
        gripper = grip_v[:, 0] if state is None else _resample_onto(timestamps, grip_t, grip_v)[:, 0]

    return Trajectory(
        timestamps_s=timestamps,
        state=state,
        state_labels=state_labels,
        action=action,
        action_labels=None,
        gripper_position=gripper,
        # Neither JointState nor Twist carries action-space semantics --
        # undeclared, honestly, until a topic/dataset convention states one.
        action_space=None,
    )
