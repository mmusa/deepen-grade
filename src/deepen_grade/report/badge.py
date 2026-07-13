# SPDX-License-Identifier: Apache-2.0
"""The funnel: what deepen-grade deliberately can't tell you locally, and the
claimable "Deepen Verified" badge for datasets that pass.
"""

from __future__ import annotations

EVALUATE_URL = "https://evaluate.deepen.ai"

FUNNEL_ITEMS: list[str] = [
    "Whether camera calibration is geometrically CORRECT, not just present -- "
    "targetless verification (reprojection/epipolar consistency, drift-over-session) "
    "requires opening the video and is Deepen's patented, sealed audit.",
    "A signed, reproducible acceptance certificate for a data purchase or handoff.",
    "Entry in the public Robot Dataset Quality Index leaderboard.",
    "Influence-function trainability scoring (which episodes actually help your policy).",
]


def funnel_text() -> str:
    lines = [f"  - {item}" for item in FUNNEL_ITEMS]
    return (
        "What this can't tell you locally:\n"
        + "\n".join(lines)
        + f"\n\nRun the full sealed audit: {EVALUATE_URL}"
    )


def badge_markdown(dataset_source: str) -> str:
    """Markdown snippet a dataset author can paste into their HuggingFace
    dataset card / README after a passing `deepen grade` run."""
    return (
        f"[![Deepen Verified](https://img.shields.io/badge/Deepen-Verified-2a9d8f)]"
        f"({EVALUATE_URL}?src={dataset_source})\n\n"
        "Graded with [deepen-grade](https://github.com/deepen-ai/deepen-grade) -- "
        "hygiene, episode quality, and calibration sanity, every metric cited to a "
        f"published source. For a full calibration-verified acceptance audit, see {EVALUATE_URL}."
    )
