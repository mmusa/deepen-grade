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


@dataclass
class CheckResult:
    check_id: str
    name: str
    severity: Severity
    summary: str
    citation_keys: tuple[str, ...]
    details: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity != Severity.NOT_APPLICABLE and not self.citation_keys:
            raise ValueError(
                f"CheckResult {self.check_id!r} has no citation_keys -- every "
                "non-N/A check must cite a published source (see citations.py)."
            )
