# SPDX-License-Identifier: Apache-2.0
"""Reads LeRobotDataset (v2/v3) datasets: a local directory, or a HuggingFace
`repo-id` (downloaded on demand).

Deliberately does NOT depend on the `lerobot` training package (which pulls in
torch and friends) -- a LeRobot dataset on disk is just parquet + JSON, and
that's all deepen-grade needs to read state/action/timestamp columns. Video
files are never downloaded or decoded: every check here operates on the
recorded state/action arrays, not pixels, so skipping video keeps installs
and runs fast and keeps this tool privacy-friendly by default.

On-disk layout handled defensively across v2 (`data/chunk-*/episode_*.parquet`)
and v3 (`data/chunk-*/file-*.parquet`) by globbing `data/**/*.parquet` and
grouping rows by the `episode_index` column, which both versions carry.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deepen_grade.ingest.base import CalibrationInfo, Dataset, Episode, TopicInfo, Trajectory
from deepen_grade.ingest.exceptions import DatasetReadError, MissingOptionalDependencyError

STATE_COLUMN_CANDIDATES = ("observation.state", "state")
ACTION_COLUMN = "action"
GRIPPER_NAME_HINTS = ("gripper",)


def _require_deps():
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            feature="LeRobot datasets", extra="lerobot", packages=["pandas", "pyarrow"]
        ) from exc
    return pd


def looks_like_repo_id(value: str) -> bool:
    """Heuristic: 'org/name', no path separators beyond the one slash, not an
    existing local path. Used by the CLI's format auto-detection."""
    if Path(value).exists():
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(parts) and not value.startswith((".", "/", "~"))


def _resolve_local_root(path_or_repo_id: str, cache_dir: str | None) -> Path:
    local = Path(path_or_repo_id)
    if local.exists():
        return local
    if not looks_like_repo_id(path_or_repo_id):
        raise DatasetReadError(
            f"{path_or_repo_id!r} is neither an existing local path nor a "
            "HuggingFace repo-id of the form 'org/dataset-name'."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            feature="HuggingFace repo-ids",
            extra="lerobot",
            packages=["huggingface_hub"],
        ) from exc
    # Video files are intentionally excluded -- see module docstring.
    snapshot_path = snapshot_download(
        repo_id=path_or_repo_id,
        repo_type="dataset",
        allow_patterns=["meta/**", "data/**"],
        cache_dir=cache_dir,
    )
    return Path(snapshot_path)


def _load_frames(root: Path, pd):
    parquet_files = sorted(root.glob("data/**/*.parquet"))
    if not parquet_files:
        raise DatasetReadError(f"{root}: no parquet files found under data/ -- not a LeRobot dataset?")
    frames = [pd.read_parquet(p) for p in parquet_files]
    return pd.concat(frames, ignore_index=True)


def _stack_column(series) -> np.ndarray | None:
    """A LeRobot vector feature column loads as a pandas object column of
    per-row arrays/lists; stack it into a single (T, D) numpy array."""
    if series is None or series.empty:
        return None
    values = series.to_list()
    if values and not hasattr(values[0], "__len__"):
        return np.asarray(values, dtype=float).reshape(-1, 1)
    try:
        return np.stack([np.asarray(v, dtype=float) for v in values])
    except ValueError:
        return None  # ragged rows -- shouldn't happen in a valid dataset; degrade gracefully


def _load_calibration_sidecar(root: Path) -> dict:
    """Optional, deepen-grade-specific convention: `meta/calibration.json`
    mapping camera_id -> {intrinsics, extrinsics}. Not part of the official
    LeRobot spec (which has no calibration slot at all -- see README) but
    respected here so datasets that DO carry calibration (e.g. converted from
    a format that has it) still get sanity-checked instead of always reading
    "not present".
    """
    sidecar = root / "meta" / "calibration.json"
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _calibrations_from_sidecar(raw: dict) -> list[CalibrationInfo]:
    calibrations = []
    for cam_id, entry in raw.items():
        calibrations.append(
            CalibrationInfo(
                camera_id=cam_id,
                intrinsics=entry.get("intrinsics"),
                extrinsics=entry.get("extrinsics"),
                source="meta/calibration.json",
            )
        )
    return calibrations


def read_lerobot(path_or_repo_id: str, cache_dir: str | None = None) -> Dataset:
    pd = _require_deps()
    root = _resolve_local_root(path_or_repo_id, cache_dir)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    fps = info.get("fps", 30)
    features: dict = info.get("features", {})

    camera_features = sorted(
        name for name, spec in features.items() if spec.get("dtype") in ("video", "image")
    )

    df = _load_frames(root, pd)
    if "episode_index" not in df.columns:
        raise DatasetReadError(f"{path_or_repo_id}: no 'episode_index' column in data/*.parquet")

    state_col = next((c for c in STATE_COLUMN_CANDIDATES if c in df.columns), STATE_COLUMN_CANDIDATES[0])
    state_labels = list(features[state_col]["names"]) if features.get(state_col, {}).get("names") else None
    action_labels = (
        list(features[ACTION_COLUMN]["names"]) if features.get(ACTION_COLUMN, {}).get("names") else None
    )

    calibration_sidecar = _calibrations_from_sidecar(_load_calibration_sidecar(root))

    warnings: list[str] = []
    episodes: list[Episode] = []
    for ep_idx, group in df.groupby("episode_index"):
        sort_col = "frame_index" if "frame_index" in group.columns else "timestamp"
        group = group.sort_values(sort_col) if sort_col in group.columns else group
        episodes.append(
            _build_episode(int(ep_idx), group, fps, state_col, state_labels, action_labels,
                            camera_features, calibration_sidecar)
        )

    if not any(ep.trajectory is not None for ep in episodes):
        candidates = " / ".join((*STATE_COLUMN_CANDIDATES, ACTION_COLUMN))
        warnings.append(
            f"None of [{candidates}] found as a column in this dataset's parquet frames: "
            "episode-quality checks skipped."
        )
    if not calibration_sidecar:
        warnings.append(
            "No camera calibration metadata found (LeRobot's core format has no "
            "standard intrinsics/extrinsics slot). Calibration sanity: NOT PRESENT."
        )

    ds = Dataset(source=str(path_or_repo_id), format="lerobot", episodes=episodes)
    ds.warnings.extend(warnings)
    return ds


def _build_episode(ep_idx, group, fps, state_col, state_labels, action_labels,
                    camera_features, calibration_sidecar) -> Episode:
    if "timestamp" in group.columns:
        timestamps = group["timestamp"].to_numpy(dtype=float)
    elif "frame_index" in group.columns:
        timestamps = group["frame_index"].to_numpy(dtype=float) / float(fps)
    else:
        timestamps = np.arange(len(group), dtype=float) / float(fps)

    state = _stack_column(group[state_col]) if state_col in group.columns else None
    action = _stack_column(group[ACTION_COLUMN]) if ACTION_COLUMN in group.columns else None
    gripper = _extract_gripper(state, state_labels, action, action_labels)

    trajectory = None
    if state is not None or action is not None:
        trajectory = Trajectory(
            timestamps_s=timestamps,
            state=state,
            state_labels=state_labels,
            action=action,
            action_labels=action_labels,
            gripper_position=gripper,
        )

    topics = [
        TopicInfo(name=state_col, msg_type="observation.state", count=len(group), timestamps_s=timestamps)
    ] if state is not None else []
    if action is not None:
        topics.append(TopicInfo(name=ACTION_COLUMN, msg_type="action", count=len(group), timestamps_s=timestamps))
    for cam in camera_features:
        topics.append(TopicInfo(name=cam, msg_type="video", count=len(group), timestamps_s=timestamps))

    task = None
    if "task" in group.columns and len(group):
        task = str(group["task"].iloc[0])
    elif "task_index" in group.columns and len(group):
        task = f"task_index={int(group['task_index'].iloc[0])}"

    episode_id = f"episode_{ep_idx:06d}"
    return Episode(
        episode_id=episode_id,
        topics=topics,
        tf_edges=None,  # LeRobot has no tf concept
        trajectory=trajectory,
        calibrations=calibration_sidecar,  # dataset-level; same for every episode
        duration_s=float(timestamps[-1] - timestamps[0]) if len(timestamps) else None,
        task=task,
    )


def _extract_gripper(state, state_labels, action, action_labels) -> np.ndarray | None:
    for arr, labels in ((state, state_labels), (action, action_labels)):
        if arr is None or not labels:
            continue
        for i, name in enumerate(labels):
            if any(hint in name.lower() for hint in GRIPPER_NAME_HINTS) and i < arr.shape[1]:
                return arr[:, i]
    return None
