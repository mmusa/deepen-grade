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

Two more layout realities, both found grading the top-30 most-downloaded
LeRobot repos on the Hub:

- **Container repos.** Many repos aren't a single LeRobot dataset at the root
  -- they nest one or more complete LeRobot datasets one or two directories
  down (a per-contributor dir, an `ImitationLearning/` wrapper, ...). See
  `_discover_subdataset_roots`.
- **Corpus size.** Some repos have 100k+ episodes; downloading and
  DataFrame-concatenating the whole thing OOMs a laptop long before grading
  starts. Episodes are read one parquet file at a time (never concatenated
  into one corpus-sized frame) and `--sample`/`--max-episodes` are resolved
  from a cheap episode-index-only pass *before* any full file is read, so an
  unsampled run touches every file but a sampled run only touches the ones
  it needs. See `_read_single_root`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from deepen_grade.ingest.base import CalibrationInfo, Dataset, Episode, TopicInfo, Trajectory
from deepen_grade.ingest.exceptions import DatasetReadError, MissingOptionalDependencyError

STATE_COLUMN_CANDIDATES = ("observation.state", "state")
ACTION_COLUMN = "action"
GRIPPER_NAME_HINTS = ("gripper",)

# Fetch nested (container-repo) layouts too, up to two directories down --
# see module docstring. Video is deliberately never matched at any depth.
_HF_ALLOW_PATTERNS = [
    "meta/**", "data/**",
    "*/meta/**", "*/data/**",
    "*/*/meta/**", "*/*/data/**",
]

_DOWNLOAD_MAX_RETRIES = 3
_DOWNLOAD_BACKOFF_S = 2.0  # doubles each retry: 2s, 4s, 8s


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


def _resolve_local_root(path_or_repo_id: str, cache_dir: str | None, hf_token: str | None = None) -> Path:
    local = Path(path_or_repo_id)
    if local.exists():
        return local
    if not looks_like_repo_id(path_or_repo_id):
        raise DatasetReadError(
            f"{path_or_repo_id!r} is neither an existing local path nor a "
            "HuggingFace repo-id of the form 'org/dataset-name'."
        )
    return Path(_download_with_retries(path_or_repo_id, cache_dir, hf_token))


def _download_with_retries(repo_id: str, cache_dir: str | None, hf_token: str | None) -> str:
    """`snapshot_download`, retried on transient failures.

    Two real failure modes from grading Hub repos anonymously: a dropped
    connection mid-download raises `IncompleteSnapshotError` on the next
    attempt, and enough parallel graders hitting the Hub trips its anonymous
    rate limit (429) -- which without a token then sits in silent backoff.
    Both `IncompleteSnapshotError` and a non-429 `HfHubHTTPError` get up to
    `_DOWNLOAD_MAX_RETRIES` attempts with exponential backoff; a 429 fails
    immediately with an actionable message instead of retrying into the same
    wall, since a token is the actual fix.
    """
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError, IncompleteSnapshotError
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            feature="HuggingFace repo-ids", extra="lerobot", packages=["huggingface_hub"],
        ) from exc

    last_exc: Exception | None = None
    for attempt in range(_DOWNLOAD_MAX_RETRIES):
        try:
            return snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=_HF_ALLOW_PATTERNS,
                cache_dir=cache_dir,
                token=hf_token,
            )
        except HfHubHTTPError as exc:
            if getattr(exc.response, "status_code", None) == 429:
                raise DatasetReadError(
                    f"{repo_id}: HuggingFace Hub rate-limited this download (HTTP 429). "
                    "Authenticate to raise your rate limit: set HF_TOKEN or pass --hf-token."
                ) from exc
            last_exc = exc
        except IncompleteSnapshotError as exc:
            last_exc = exc
        if attempt < _DOWNLOAD_MAX_RETRIES - 1:
            time.sleep(_DOWNLOAD_BACKOFF_S * (2**attempt))

    raise DatasetReadError(
        f"{repo_id}: download failed after {_DOWNLOAD_MAX_RETRIES} attempts "
        f"({last_exc.__class__.__name__}: {last_exc})"
    ) from last_exc


def _discover_subdataset_roots(root: Path) -> list[Path]:
    """`[root]` if `root` itself is a LeRobot dataset; otherwise every nested
    dataset root found up to two directories down (glob depth <= 3), sorted
    for determinism. Empty if neither -- caller decides that's an error.
    """
    if (root / "meta" / "info.json").exists() or (root / "data").exists():
        return [root]
    nested = {p.parent.parent for p in root.glob("*/meta/info.json")}
    nested |= {p.parent.parent for p in root.glob("*/*/meta/info.json")}
    return sorted(nested)


def looks_like_lerobot_container(root: Path) -> bool:
    """Cheap existence check used by `detect.py` before a full read: does
    `root` nest LeRobot dataset(s) even though it isn't one itself?"""
    return bool(_discover_subdataset_roots(root))


def _one_line_reason(exc: Exception) -> str:
    """A single-line, class-prefixed summary of a caught exception -- used to
    report a failed sub-dataset in a container without a stack trace. A
    `DatasetReadError` already reads clearly on its own (its message is the
    fix); anything else gets its class name prefixed for context, since e.g.
    a bare `AttributeError: 'list' object has no attribute 'lower'` is
    meaningless without knowing what raised it.
    """
    text = str(exc).splitlines()[0] if str(exc) else "(no message)"
    if isinstance(exc, DatasetReadError):
        return text
    return f"{exc.__class__.__name__}: {text}"


def _normalize_labels(raw) -> list[str] | None:
    """`features[col]["names"]` is documented as a flat `list[str]`, but some
    real repos store it as a single-element nested list instead (e.g. a
    grouped-joint layout: `[[j0, j1, ..., j25]]`). Unwrap exactly that one
    unambiguous shape; anything stranger (multi-group nesting, non-string
    entries) degrades to `None` rather than crashing `_extract_gripper` or
    any other labels[i] consumer downstream.
    """
    if not raw or not isinstance(raw, list):
        return None
    if len(raw) == 1 and isinstance(raw[0], list):
        raw = raw[0]
    if not raw or not all(isinstance(x, (str, int, float)) for x in raw):
        return None
    return [str(x) for x in raw]


def _labels_matching(labels: list[str] | None, arr: np.ndarray | None) -> list[str] | None:
    """A label list whose length doesn't match the array it's meant to label
    (e.g. a normalized-but-still-wrong-length names list) is worse than no
    labels at all -- degrade to `None` so downstream code falls back to its
    own "dim_N" naming instead of mis-attributing a label to the wrong column.
    """
    if labels is None or arr is None or len(labels) == arr.shape[1]:
        return labels
    return None


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


def _calibrations_from_mapping(raw: dict, source: str) -> list[CalibrationInfo]:
    return [
        CalibrationInfo(
            camera_id=str(cam_id),
            intrinsics=entry.get("intrinsics"),
            extrinsics=entry.get("extrinsics"),
            source=source,
        )
        for cam_id, entry in raw.items()
    ]


def _calibrations_from_droid_style(root: Path, info: dict) -> list[CalibrationInfo]:
    """Best-effort read of a DROID-style calibration convention, narrowly
    scoped to two obvious shapes -- not a full DROID schema implementation.

    None of the top-30-most-downloaded LeRobot repos on the Hub actually
    carried calibration metadata (see README), so there's no real corpus to
    validate a fuller parser against yet; this only claims the two shapes
    that are unambiguous to interpret:
      1. `meta/info.json["calibration"]`, shaped like our own sidecar
         convention (camera_id -> {intrinsics, extrinsics}).
      2. `meta/cam2base_extrinsics.json`: camera_id -> a 6-vector
         [x,y,z,rx,ry,rz] (or a flattened 4x4) base-frame extrinsic. No
         intrinsics claim is made from this file, so calibration_sanity will
         (correctly) flag "intrinsics missing" for cameras only found here --
         that's an accurate signal, not a bug.
    """
    inline = info.get("calibration")
    if isinstance(inline, dict) and inline:
        return _calibrations_from_mapping(inline, source="meta/info.json:calibration")

    cam2base_path = root / "meta" / "cam2base_extrinsics.json"
    if not cam2base_path.exists():
        return []
    try:
        raw = json.loads(cam2base_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(raw, dict):
        return []
    return [
        CalibrationInfo(
            camera_id=str(cam_id),
            intrinsics=None,
            extrinsics=[float(v) for v in vec] if isinstance(vec, list) else None,
            source="meta/cam2base_extrinsics.json",
        )
        for cam_id, vec in raw.items()
    ]


def _select_episode_ids(
    all_ids: list[int], sample: int | None, seed: int, max_episodes: int | None
) -> tuple[list[int], dict | None]:
    """Which episode_index values to materialize, and the `sampling` block
    (None if neither --sample nor --max-episodes is active) to surface in the
    report. `sample` draws uniformly without replacement, seeded for
    determinism; `max_episodes` takes the first N in natural (sorted) order.
    """
    total = len(all_ids)
    if sample is not None:
        n = min(sample, total)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(np.asarray(all_ids), size=n, replace=False) if n else np.asarray([], dtype=int)
        return sorted(int(x) for x in chosen), {
            "mode": "sample", "n": n, "seed": seed, "episodes_total": total,
        }
    if max_episodes is not None:
        n = min(max_episodes, total)
        return sorted(all_ids)[:n], {"mode": "head", "n": n, "seed": None, "episodes_total": total}
    return list(all_ids), None


def _episode_ids_by_file(parquet_files: list[Path], pd) -> dict[Path, np.ndarray]:
    """Cheap first pass: read only the `episode_index` column of each file so
    the full episode universe (and the --sample/--max-episodes selection) is
    known before any full-file materialization -- the whole point of
    --sample is to avoid touching most of the corpus at all.
    """
    return {p: pd.read_parquet(p, columns=["episode_index"])["episode_index"].unique() for p in parquet_files}


def _peek_columns(path: Path) -> set[str]:
    """Column names of a parquet file without reading any row data."""
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema_arrow.names)


def _read_single_root(
    root: Path, pd, sample: int | None, seed: int, max_episodes: int | None
) -> tuple[list[Episode], list[str], dict | None]:
    """Read one LeRobot dataset root, bounding peak memory to one parquet
    file at a time (never the whole corpus -- see module docstring)."""
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    fps = info.get("fps", 30)
    features: dict = info.get("features", {})
    camera_features = sorted(
        name for name, spec in features.items() if spec.get("dtype") in ("video", "image")
    )

    parquet_files = sorted(root.glob("data/**/*.parquet"))
    if not parquet_files:
        raise DatasetReadError(f"{root}: no parquet files found under data/ -- not a LeRobot dataset?")

    columns = _peek_columns(parquet_files[0])
    if "episode_index" not in columns:
        raise DatasetReadError(f"{root}: no 'episode_index' column in data/*.parquet")
    state_col = next((c for c in STATE_COLUMN_CANDIDATES if c in columns), STATE_COLUMN_CANDIDATES[0])
    state_labels = _normalize_labels(features.get(state_col, {}).get("names"))
    action_labels = _normalize_labels(features.get(ACTION_COLUMN, {}).get("names"))

    calibration = _calibrations_from_mapping(
        _load_calibration_sidecar(root), source="meta/calibration.json"
    ) or _calibrations_from_droid_style(root, info)

    episode_ids_by_file = _episode_ids_by_file(parquet_files, pd)
    all_episode_ids = sorted({int(eid) for ids in episode_ids_by_file.values() for eid in ids})
    selected, sampling = _select_episode_ids(all_episode_ids, sample, seed, max_episodes)
    selected_set = set(selected)

    episodes: list[Episode] = []
    for path in parquet_files:
        if not any(int(eid) in selected_set for eid in episode_ids_by_file[path]):
            continue  # nothing selected lives in this file -- never read it
        df = pd.read_parquet(path)  # bounded to this one file's rows
        for ep_idx, group in df.groupby("episode_index"):
            if int(ep_idx) not in selected_set:
                continue
            sort_col = "frame_index" if "frame_index" in group.columns else "timestamp"
            group = group.sort_values(sort_col) if sort_col in group.columns else group
            episodes.append(
                _build_episode(int(ep_idx), group, fps, state_col, state_labels, action_labels,
                                camera_features, calibration)
            )
        del df  # next file's frame replaces this one -- corpus is never held whole

    episodes.sort(key=lambda ep: ep.episode_id)

    warnings: list[str] = []
    if not any(ep.trajectory is not None for ep in episodes):
        candidates = " / ".join((*STATE_COLUMN_CANDIDATES, ACTION_COLUMN))
        warnings.append(
            f"None of [{candidates}] found as a column in this dataset's parquet frames: "
            "episode-quality checks skipped."
        )
    if not calibration:
        warnings.append(
            "No camera calibration metadata found (LeRobot's core format has no "
            "standard intrinsics/extrinsics slot). Calibration sanity: NOT PRESENT."
        )
    return episodes, warnings, sampling


def _merge_sampling(parts: list[dict]) -> dict:
    """Combine per-subdataset sampling blocks (container repos apply
    --sample/--max-episodes independently within each nested dataset) into
    one report-level summary. Mode is kept only if every part agrees;
    otherwise reported as "mixed" rather than silently picking one.
    """
    modes = {p["mode"] for p in parts}
    seeds = {p["seed"] for p in parts}
    return {
        "mode": modes.pop() if len(modes) == 1 else "mixed",
        "n": sum(p["n"] for p in parts),
        "seed": seeds.pop() if len(seeds) == 1 else None,
        "episodes_total": sum(p["episodes_total"] for p in parts),
    }


def read_lerobot(
    path_or_repo_id: str,
    cache_dir: str | None = None,
    subdataset: str | None = None,
    sample: int | None = None,
    seed: int = 0,
    max_episodes: int | None = None,
    hf_token: str | None = None,
) -> Dataset:
    pd = _require_deps()
    root = _resolve_local_root(path_or_repo_id, cache_dir, hf_token=hf_token)

    sub_roots = _discover_subdataset_roots(root)
    if not sub_roots:
        raise DatasetReadError(
            f"{root}: no meta/info.json at the root, and none found nested one or two "
            "directories down either -- not a LeRobot dataset or container repo?"
        )
    is_container = not (len(sub_roots) == 1 and sub_roots[0] == root)

    if subdataset is not None:
        target = (root / subdataset).resolve()
        matches = [p for p in sub_roots if p.resolve() == target]
        if not matches:
            available = ", ".join(str(p.relative_to(root)) for p in sub_roots) or "(none found)"
            raise DatasetReadError(f"--subdataset {subdataset!r} not found under {root}. Available: {available}")
        sub_roots = matches
        is_container = False  # an explicit single pick behaves like a normal single dataset

    all_episodes: list[Episode] = []
    all_warnings: list[str] = []
    sampling_parts: list[dict] = []
    failures: list[str] = []
    for sub_root in sub_roots:
        prefix = str(sub_root.relative_to(root)) if is_container else None
        try:
            episodes, warnings, sampling = _read_single_root(sub_root, pd, sample, seed, max_episodes)
        except Exception as exc:  # noqa: BLE001 -- one broken sub-dataset must not sink the whole container
            if not is_container:
                raise  # single-dataset (or explicit --subdataset) path: fail exactly as before
            reason = _one_line_reason(exc)
            failures.append(f"{prefix}: {reason}")
            all_warnings.append(f"sub-dataset {prefix} could not be read: {reason}")
            continue
        if prefix:
            for ep in episodes:
                ep.episode_id = f"{prefix}::{ep.episode_id}"
        all_episodes.extend(episodes)
        all_warnings.extend(warnings)
        if sampling:
            sampling_parts.append(sampling)

    if is_container and failures and len(failures) == len(sub_roots):
        raise DatasetReadError(
            f"{root}: every nested sub-dataset failed to read -- " + "; ".join(failures)
        )

    if is_container:
        names = ", ".join(str(p.relative_to(root)) for p in sub_roots)
        all_warnings.insert(
            0,
            f"container repo: {len(sub_roots)} nested LeRobot dataset(s) found and graded "
            f"together ({names}). Episode IDs are prefixed 'subpath::episode_id'. "
            "Use --subdataset to grade just one.",
        )

    ds = Dataset(source=str(path_or_repo_id), format="lerobot", episodes=all_episodes)
    ds.warnings.extend(all_warnings)
    if sampling_parts:
        ds.sampling = sampling_parts[0] if len(sampling_parts) == 1 else _merge_sampling(sampling_parts)
    return ds


def _build_episode(ep_idx, group, fps, state_col, state_labels, action_labels,
                    camera_features, calibration) -> Episode:
    if "timestamp" in group.columns:
        timestamps = group["timestamp"].to_numpy(dtype=float)
    elif "frame_index" in group.columns:
        timestamps = group["frame_index"].to_numpy(dtype=float) / float(fps)
    else:
        timestamps = np.arange(len(group), dtype=float) / float(fps)

    state = _stack_column(group[state_col]) if state_col in group.columns else None
    action = _stack_column(group[ACTION_COLUMN]) if ACTION_COLUMN in group.columns else None
    state_labels = _labels_matching(state_labels, state)
    action_labels = _labels_matching(action_labels, action)
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
            # LeRobot's core spec has no action-space-semantics slot (see
            # GRADING_TAXONOMY_V1.md's gap table) -- undeclared, honestly, until
            # a format or a dataset's own metadata actually states one.
            action_space=None,
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
        calibrations=calibration,  # dataset-level; same for every episode
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
