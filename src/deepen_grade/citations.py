# SPDX-License-Identifier: Apache-2.0
"""Central registry of the published sources every check in deepen-grade traces to.

deepen-grade's rule is simple: no number in a report exists without a citation a
reader can go verify. Hygiene checks cite the open standards that define what
"correct" means (REP 105, ASAM OpenLABEL, the ISO/WD 26264-1 companion paper).
Episode-quality checks cite the papers that defined the metric (log-dimensionless
jerk, DQAF, SCIZOR). The calibration sanity check cites the paper that shows why
calibration correctness matters -- it deliberately does NOT cite a verification
method, because deepen-grade does not ship one (see README: "What this can't
tell you locally").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    key: str
    title: str
    authors: str
    venue: str
    url: str

    def short(self) -> str:
        return f"{self.authors} -- {self.title} ({self.venue})"


REP_105 = Citation(
    key="rep105",
    title="Coordinate Frames for Mobile Platforms",
    authors="Meeussen, W.",
    venue="ROS Enhancement Proposal 105",
    url="https://www.ros.org/reps/rep-0105.html",
)

DATA_STANDARDS_HUMANOID = Citation(
    key="humanoid-data-standards",
    title="Data Standards for Humanoid Robotics",
    authors="Various (ISO/WD 26264-1 companion)",
    venue="arXiv:2606.19769",
    url="https://arxiv.org/pdf/2606.19769",
)

DQAF = Citation(
    key="dqaf",
    title="Closing the Loop in Teleoperation: Episode-Level Data Quality Assessment",
    authors="Siemens",
    venue="arXiv:2605.26349",
    url="https://arxiv.org/html/2605.26349v1",
)

SCIZOR = Citation(
    key="scizor",
    title="SCIZOR: Self-Supervised Data Curation for Large-Scale Imitation Learning",
    authors="Various",
    venue="arXiv:2505.22626",
    url="https://arxiv.org/abs/2505.22626",
)

LDJ_SMOOTHNESS = Citation(
    key="ldj",
    title="A Robust and Sensitive Metric for Quantifying Movement Smoothness",
    authors="Balasubramanian, Melendez-Calderon, Roby-Brami, Burdet",
    venue="IEEE Transactions on Biomedical Engineering, 62(8), 2015",
    url="https://doi.org/10.1109/TBME.2015.2404082",
)

SHADOW_EXTRINSICS_NOISE = Citation(
    key="shadow",
    title="Shadow: policy performance degrades proportionally with extrinsics noise",
    authors="Various",
    venue="arXiv:2503.00774",
    url="https://arxiv.org/pdf/2503.00774",
)

ASAM_OPENLABEL = Citation(
    key="openlabel",
    title="ASAM OpenLABEL -- open standard for labeling and annotating sensor data",
    authors="ASAM e.V.",
    venue="ASAM OpenLABEL Specification",
    url="https://www.asam.net/standards/detail/openlabel/",
)

ALL_CITATIONS: dict[str, Citation] = {
    c.key: c
    for c in (
        REP_105,
        DATA_STANDARDS_HUMANOID,
        DQAF,
        SCIZOR,
        LDJ_SMOOTHNESS,
        SHADOW_EXTRINSICS_NOISE,
        ASAM_OPENLABEL,
    )
}
