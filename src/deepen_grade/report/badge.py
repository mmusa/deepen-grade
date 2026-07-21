# SPDX-License-Identifier: Apache-2.0
"""The funnel: what deepen-grade deliberately can't tell you locally, and the
self-assessment framing for what it *can*.

There is deliberately no "Deepen Verified" badge here. deepen-grade runs on
your machine, on your data, with nothing checked by Deepen -- so its output is
a SELF-ASSESSMENT, and a trust mark Deepen never saw would be an honor-system
claim. Signed, third-party attestation exists only as the sealed deep audit.
"""

from __future__ import annotations

EVALUATE_URL = "https://deepen-robograde.pages.dev"

SELF_ASSESSMENT_NOTE = (
    "This is a local SELF-ASSESSMENT: it was produced on your machine and is "
    "not reviewed, signed, or attested by Deepen. Do not present it as a "
    "Deepen verification. For a signed, third-party acceptance audit, see "
    f"{EVALUATE_URL}."
)

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
