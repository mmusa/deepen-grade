# SPDX-License-Identifier: Apache-2.0
"""deepen-grade: a free, local CLI that grades robot-learning datasets on
hygiene, episode quality, and calibration sanity -- every metric traced to a
named paper. See https://github.com/deepen-ai/deepen-grade.
"""

from deepen_grade.__about__ import __version__
from deepen_grade.grading import grade_dataset, passes_verification
from deepen_grade.ingest import load_dataset

__all__ = ["__version__", "load_dataset", "grade_dataset", "passes_verification"]
