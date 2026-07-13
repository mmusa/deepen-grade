# SPDX-License-Identifier: Apache-2.0
"""Ingestion: adapters that normalize .mcap / .bag / .db3 / LeRobot sources
into the common `deepen_grade.ingest.base.Dataset` model that every check
operates on. `load_dataset` is the one entry point the CLI needs."""

from deepen_grade.ingest.base import (
    CalibrationInfo,
    Dataset,
    Episode,
    TfEdge,
    TopicInfo,
    Trajectory,
)
from deepen_grade.ingest.detect import detect_format, load_dataset

__all__ = [
    "CalibrationInfo",
    "Dataset",
    "Episode",
    "TfEdge",
    "TopicInfo",
    "Trajectory",
    "detect_format",
    "load_dataset",
]
