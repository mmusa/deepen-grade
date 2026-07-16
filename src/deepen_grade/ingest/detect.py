# SPDX-License-Identifier: Apache-2.0
"""Format auto-detection and the single dispatch entry point, `load_dataset`.

No config required for the common case: `deepen grade <path>` just works by
looking at the extension (or, for a bare directory/repo-id, at its shape).
"""

from __future__ import annotations

from pathlib import Path

from deepen_grade.ingest.base import Dataset
from deepen_grade.ingest.exceptions import UnsupportedFormatError

_EXTENSION_FORMATS = {
    ".mcap": "mcap",
    ".bag": "ros1_bag",
    ".db3": "ros2_db3",
}


def detect_format(path_or_repo_id: str) -> str:
    """Return one of: mcap, ros1_bag, ros2_db3, lerobot."""
    p = Path(path_or_repo_id)

    if p.is_file():
        fmt = _EXTENSION_FORMATS.get(p.suffix)
        if fmt is None:
            raise UnsupportedFormatError(
                f"{path_or_repo_id}: unrecognized file extension {p.suffix!r}. "
                "deepen-grade reads .mcap, .bag (ROS1), .db3 (ROS2), or a LeRobot "
                "dataset directory / HuggingFace repo-id."
            )
        return fmt

    if p.is_dir():
        if (p / "meta" / "info.json").exists() or (p / "data").exists():
            return "lerobot"
        if (p / "metadata.yaml").exists():
            return "ros2_db3"
        from deepen_grade.ingest.lerobot_reader import looks_like_lerobot_container

        if looks_like_lerobot_container(p):
            return "lerobot"
        raise UnsupportedFormatError(
            f"{path_or_repo_id}: directory doesn't look like a LeRobot dataset "
            "(no meta/info.json or data/), a container of nested LeRobot datasets, "
            "or a ROS2 bag directory (no metadata.yaml)."
        )

    # Not an existing local path at all -- only a HuggingFace repo-id is valid here.
    from deepen_grade.ingest.lerobot_reader import looks_like_repo_id

    if looks_like_repo_id(path_or_repo_id):
        return "lerobot"

    raise UnsupportedFormatError(
        f"{path_or_repo_id!r}: no such file or directory, and it doesn't look like "
        "a HuggingFace repo-id ('org/dataset-name')."
    )


def load_dataset(
    path_or_repo_id: str,
    fmt: str | None = None,
    cache_dir: str | None = None,
    subdataset: str | None = None,
    sample: int | None = None,
    seed: int = 0,
    max_episodes: int | None = None,
    hf_token: str | None = None,
) -> Dataset:
    """Load and normalize a dataset. Auto-detects format unless `fmt` is given.

    `subdataset`/`sample`/`seed`/`max_episodes`/`hf_token` are LeRobot-only
    concepts (container repos, corpus-scale sampling, HF auth) and are
    ignored for the other formats.
    """
    fmt = fmt or detect_format(path_or_repo_id)

    if fmt == "mcap":
        from deepen_grade.ingest.mcap_reader import read_mcap

        return read_mcap(path_or_repo_id)
    if fmt == "ros1_bag":
        from deepen_grade.ingest.ros_reader import read_ros1_bag

        return read_ros1_bag(path_or_repo_id)
    if fmt == "ros2_db3":
        from deepen_grade.ingest.ros_reader import read_ros2_db3

        return read_ros2_db3(path_or_repo_id)
    if fmt == "lerobot":
        from deepen_grade.ingest.lerobot_reader import read_lerobot

        return read_lerobot(
            path_or_repo_id, cache_dir=cache_dir, subdataset=subdataset, sample=sample,
            seed=seed, max_episodes=max_episodes, hf_token=hf_token,
        )

    raise UnsupportedFormatError(f"Unknown format {fmt!r}")
