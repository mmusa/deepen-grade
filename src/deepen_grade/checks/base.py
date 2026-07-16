# SPDX-License-Identifier: Apache-2.0
"""The shared result type every check returns.

Every `CheckResult` carries `citation_keys` pointing into
`deepen_grade.citations.ALL_CITATIONS` -- the report renderers refuse to
print a check that doesn't cite anything, which is the mechanical enforcement
of deepen-grade's "every number traces to a paper" rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    PASS = "pass"
    INFO = "info"
    WARN = "warn"
    FAIL = "fail"
    NOT_APPLICABLE = "n/a"


# Check IDs allowed to report a non-N/A severity without a citation -- a narrow,
# explicit exemption, never a general escape hatch. Currently just
# plausibility.robot_data, whose INFO firing is a deliberately uncited content
# heuristic (no published metric defines "does this look like robot data" the
# way DQAF/SCIZOR/LDJ define stall/smoothness/consistency) -- see
# checks/plausibility.py's module docstring.
UNCITED_CHECK_IDS: frozenset[str] = frozenset({"plausibility.robot_data"})


@dataclass
class CheckResult:
    check_id: str
    name: str
    severity: Severity
    summary: str
    citation_keys: tuple[str, ...]
    details: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.severity != Severity.NOT_APPLICABLE
            and not self.citation_keys
            and self.check_id not in UNCITED_CHECK_IDS
        ):
            raise ValueError(
                f"CheckResult {self.check_id!r} has no citation_keys -- every "
                "non-N/A check must cite a published source (see citations.py)."
            )
