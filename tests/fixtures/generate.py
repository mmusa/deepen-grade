# SPDX-License-Identifier: Apache-2.0
"""Generates the static test fixtures under tests/fixtures/data/ from the
shared scenario in scenario.py. Run `python tests/fixtures/generate.py` to
regenerate after changing the scenario. Fixtures are checked in as small
binary artifacts so tests don't depend on optional extras being installed
just to build them (only to *read* the ones relevant to that test).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from scenario import JOINT_LABELS, build_scenario

DATA_DIR = Path(__file__).parent / "data"
T0_NS = 1_700_000_000_000_000_000


def write_mcap(path: Path) -> None:
    from mcap_ros2.writer import Writer as McapWriter

    sc = build_scenario()
    with open(path, "wb") as f:
        writer = McapWriter(f)
        js_schema = writer.register_msgdef(
            "sensor_msgs/msg/JointState",
            "string[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort\n",
        )
        img_schema = writer.register_msgdef("sensor_msgs/msg/CompressedImage", "uint8[] data\n")

        for i, t in enumerate(sc.t):
            writer.write_message(
                "/joint_states", js_schema,
                {"name": JOINT_LABELS, "position": sc.joint_pos[i].tolist(),
                 "velocity": [0.0] * 3, "effort": [0.0] * 3},
                log_time=int(T0_NS + t * 1e9),
            )
            writer.write_message(
                "/joint_command", js_schema,
                {"name": JOINT_LABELS, "position": sc.action_pos[i].tolist(),
                 "velocity": [0.0] * 3, "effort": [0.0] * 3},
                log_time=int(T0_NS + t * 1e9),
            )
            writer.write_message(
                "/gripper/joint_states", js_schema,
                {"name": ["gripper"], "position": [float(sc.gripper_pos[i])],
                 "velocity": [0.0], "effort": [0.0]},
                log_time=int(T0_NS + sc.gripper_t[i] * 1e9),
            )
        for t in sc.camera_t[::2]:  # ~10Hz camera vs 20Hz control loop
            writer.write_message("/camera/image_raw", img_schema, {"data": [0, 1, 2]},
                                  log_time=int(T0_NS + t * 1e9))
        writer.finish()


def _ros_typestore(store):
    from rosbags.typesys import get_typestore

    return get_typestore(store)


def _header(ts):
    """std_msgs/Header differs between ROS1 (has `seq`) and ROS2 (doesn't)."""
    import dataclasses

    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    kwargs = {"stamp": Time(sec=0, nanosec=0), "frame_id": ""}
    if "seq" in {f.name for f in dataclasses.fields(Header)}:
        kwargs["seq"] = 0
    return Header(**kwargs)


def _joint_state_msg(ts, names, position, velocity, effort):
    JointState = ts.types["sensor_msgs/msg/JointState"]
    return JointState(
        header=_header(ts), name=names, position=np.asarray(position),
        velocity=np.asarray(velocity), effort=np.asarray(effort),
    )


def write_ros1_bag(path: Path) -> None:
    from rosbags.rosbag1 import Writer as Ros1Writer
    from rosbags.typesys import Stores

    ts = _ros_typestore(Stores.ROS1_NOETIC)
    sc = build_scenario()
    if path.exists():
        path.unlink()
    with Ros1Writer(path) as writer:
        conn_state = writer.add_connection("/joint_states", "sensor_msgs/msg/JointState", typestore=ts)
        conn_action = writer.add_connection("/joint_command", "sensor_msgs/msg/JointState", typestore=ts)
        conn_grip = writer.add_connection("/gripper/joint_states", "sensor_msgs/msg/JointState", typestore=ts)
        conn_cam = writer.add_connection("/camera/image_raw", "sensor_msgs/msg/CompressedImage", typestore=ts)
        CompressedImage = ts.types["sensor_msgs/msg/CompressedImage"]

        for i, t in enumerate(sc.t):
            msg = _joint_state_msg(ts, JOINT_LABELS, sc.joint_pos[i], [0.0] * 3, [0.0] * 3)
            writer.write(conn_state, int(T0_NS + t * 1e9), ts.serialize_ros1(msg, "sensor_msgs/msg/JointState"))
            msg = _joint_state_msg(ts, JOINT_LABELS, sc.action_pos[i], [0.0] * 3, [0.0] * 3)
            writer.write(conn_action, int(T0_NS + t * 1e9), ts.serialize_ros1(msg, "sensor_msgs/msg/JointState"))
            msg = _joint_state_msg(ts, ["gripper"], [float(sc.gripper_pos[i])], [0.0], [0.0])
            writer.write(conn_grip, int(T0_NS + sc.gripper_t[i] * 1e9),
                         ts.serialize_ros1(msg, "sensor_msgs/msg/JointState"))
        for t in sc.camera_t[::2]:
            img = CompressedImage(header=_header(ts), format="jpeg",
                                   data=np.array([0, 1, 2], dtype=np.uint8))
            writer.write(conn_cam, int(T0_NS + t * 1e9),
                         ts.serialize_ros1(img, "sensor_msgs/msg/CompressedImage"))


def write_ros2_bag(dir_path: Path) -> None:
    from rosbags.rosbag2 import Writer as Ros2Writer
    from rosbags.typesys import Stores

    ts = _ros_typestore(Stores.ROS2_HUMBLE)
    sc = build_scenario()
    if dir_path.exists():
        shutil.rmtree(dir_path)
    with Ros2Writer(dir_path, version=Ros2Writer.VERSION_LATEST) as writer:
        conn_state = writer.add_connection("/joint_states", "sensor_msgs/msg/JointState", typestore=ts)
        conn_action = writer.add_connection("/joint_command", "sensor_msgs/msg/JointState", typestore=ts)
        conn_grip = writer.add_connection("/gripper/joint_states", "sensor_msgs/msg/JointState", typestore=ts)

        for i, t in enumerate(sc.t):
            msg = _joint_state_msg(ts, JOINT_LABELS, sc.joint_pos[i], [0.0] * 3, [0.0] * 3)
            writer.write(conn_state, int(T0_NS + t * 1e9), ts.serialize_cdr(msg, "sensor_msgs/msg/JointState"))
            msg = _joint_state_msg(ts, JOINT_LABELS, sc.action_pos[i], [0.0] * 3, [0.0] * 3)
            writer.write(conn_action, int(T0_NS + t * 1e9), ts.serialize_cdr(msg, "sensor_msgs/msg/JointState"))
            msg = _joint_state_msg(ts, ["gripper"], [float(sc.gripper_pos[i])], [0.0], [0.0])
            writer.write(conn_grip, int(T0_NS + sc.gripper_t[i] * 1e9),
                         ts.serialize_cdr(msg, "sensor_msgs/msg/JointState"))


def write_lerobot(dir_path: Path) -> None:
    import pandas as pd

    sc = build_scenario()
    if dir_path.exists():
        shutil.rmtree(dir_path)
    (dir_path / "data" / "chunk-000").mkdir(parents=True)
    (dir_path / "meta").mkdir(parents=True)

    # LeRobot convention: gripper as the 4th state/action dim, named by feature `names`.
    state = np.concatenate([sc.joint_pos, sc.gripper_pos[:, None]], axis=1)
    action = np.concatenate([sc.action_pos, sc.gripper_pos[:, None]], axis=1)
    labels = [*JOINT_LABELS, "gripper"]

    n_half = len(sc.t) // 2
    for ep_idx, (lo, hi) in enumerate([(0, n_half), (n_half, len(sc.t))]):
        df = pd.DataFrame({
            "episode_index": ep_idx,
            "frame_index": np.arange(hi - lo),
            "timestamp": sc.t[lo:hi] - sc.t[lo],
            "task_index": 0,
            "observation.state": list(state[lo:hi]),
            "action": list(action[lo:hi]),
        })
        df.to_parquet(dir_path / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")

    info = {
        "fps": int(round(1.0 / (sc.t[1] - sc.t[0]))),
        "total_episodes": 2,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [len(labels)], "names": labels},
            "action": {"dtype": "float32", "shape": [len(labels)], "names": labels},
        },
    }
    (dir_path / "meta" / "info.json").write_text(json.dumps(info, indent=2))


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_mcap(DATA_DIR / "sample.mcap")
    print("wrote", DATA_DIR / "sample.mcap")
    write_ros1_bag(DATA_DIR / "sample_ros1.bag")
    print("wrote", DATA_DIR / "sample_ros1.bag")
    write_ros2_bag(DATA_DIR / "sample_ros2")
    print("wrote", DATA_DIR / "sample_ros2")
    write_lerobot(DATA_DIR / "sample_lerobot")
    print("wrote", DATA_DIR / "sample_lerobot")


if __name__ == "__main__":
    main()
