# SPDX-License-Identifier: Apache-2.0
"""Dev-only utility: convert one real DROID raw episode (trajectory.h5 +
metadata.json + intrinsics.json) into a minimal, on-disk LeRobotDataset-v3-shaped
directory that `deepen grade` can read via its normal LeRobot ingestion path.

NOT part of the deepen-grade package -- this script is only here to produce a
REAL-data validation fixture from real DROID recordings, so `deepen grade`
can be exercised against real robot trajectories and real (known-imperfect)
DROID camera calibration, end to end, offline. It intentionally does nothing
DROID/dataset-specific inside the `deepen_grade` package itself; the package
only ever sees a generic LeRobot-shaped dataset directory.

Usage:
    python scripts/droid_to_lerobot.py \
        --episode-dir /path/to/droid/episodes/<uuid> \
        --intrinsics /path/to/droid/intrinsics.json \
        --out /tmp/droid_lerobot_fixture
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

STATE_LABELS = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]


def load_episode(episode_dir: Path) -> dict:
    with h5py.File(episode_dir / "trajectory.h5") as h5:
        state = np.concatenate(
            [
                h5["observation/robot_state/joint_positions"][:],
                h5["observation/robot_state/gripper_position"][:][:, None],
            ],
            axis=1,
        )
        action = np.concatenate(
            [h5["action/joint_position"][:], h5["action/gripper_position"][:][:, None]],
            axis=1,
        )
        # DROID stores {seconds, nanos-within-that-second} separately (like
        # ROS's sec/nsec header convention) -- robot_timestamp_nanos ALONE is
        # not a monotonic absolute timestamp, it wraps every second.
        t_sec = h5["observation/timestamp/robot_state/robot_timestamp_seconds"][:].astype(np.int64)
        t_nsec = h5["observation/timestamp/robot_state/robot_timestamp_nanos"][:].astype(np.int64)
        t_ns = t_sec * 1_000_000_000 + t_nsec
    metadata = json.loads((episode_dir / "metadata.json").read_text())
    return {"state": state, "action": action, "t_ns": t_ns, "metadata": metadata}


def build_calibration_sidecar(metadata: dict, intrinsics: dict) -> dict:
    episode_id = metadata["uuid"]
    per_cam_intrinsics = intrinsics.get(episode_id, {})
    calibration = {}
    for cam_key in ("wrist_cam", "ext1_cam", "ext2_cam"):
        serial = metadata.get(f"{cam_key}_serial")
        extrinsics = metadata.get(f"{cam_key}_extrinsics")
        if serial is None:
            continue
        intr = per_cam_intrinsics.get(serial)
        calibration[serial] = {
            "intrinsics": (
                {
                    "fx": intr["cameraMatrix"][0], "fy": intr["cameraMatrix"][2],
                    "cx": intr["cameraMatrix"][1], "cy": intr["cameraMatrix"][3],
                    "width": intr["width"], "height": intr["height"],
                }
                if intr else None
            ),
            "extrinsics": extrinsics,
        }
    return calibration


def write_fixture(out_dir: Path, episode: dict, calibration: dict) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "data" / "chunk-000").mkdir(parents=True)
    (out_dir / "meta").mkdir(parents=True)

    t_s = (episode["t_ns"] - episode["t_ns"][0]) / 1e9
    n = len(t_s)
    df = pd.DataFrame({
        "episode_index": 0,
        "frame_index": np.arange(n),
        "timestamp": t_s,
        "task_index": 0,
        "observation.state": list(episode["state"]),
        "action": list(episode["action"]),
    })
    df.to_parquet(out_dir / "data" / "chunk-000" / "episode_000000.parquet")

    info = {
        "fps": int(round(1.0 / np.median(np.diff(t_s)))) if n > 1 else 15,
        "total_episodes": 1,
        "robot_type": "franka_panda (DROID)",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [len(STATE_LABELS)], "names": STATE_LABELS},
            "action": {"dtype": "float32", "shape": [len(STATE_LABELS)], "names": STATE_LABELS},
        },
        "source": f"DROID raw episode {episode['metadata']['uuid']} "
                  "(https://droid-dataset.github.io/), converted for local deepen-grade validation",
    }
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (out_dir / "meta" / "calibration.json").write_text(json.dumps(calibration, indent=2))
    print(f"wrote {out_dir} ({n} frames, task={episode['metadata'].get('current_task')!r}, "
          f"success={episode['metadata'].get('success')})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    episode = load_episode(args.episode_dir)
    intrinsics = json.loads(args.intrinsics.read_text())
    calibration = build_calibration_sidecar(episode["metadata"], intrinsics)
    write_fixture(args.out, episode, calibration)


if __name__ == "__main__":
    main()
