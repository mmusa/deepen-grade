# SPDX-License-Identifier: Apache-2.0
"""The shared result type every check returns.

Every `CheckResult` carries `citation_keys` pointing into
`deepen_grade.citations.ALL_CITATIONS` -- the report renderers refuse to
print a check that doesn't cite anything, which is the mechanical enforcement
of deepen-grade's "every number traces to a paper" rule.

Every `CheckResult` also carries a `claim_type` -- see `ClaimType` below. This
is the grading-taxonomy typing (GRADING_TAXONOMY_V1.md) that decides whether a
finding is even allowed to move the letter grade. `severity` (PASS/WARN/FAIL/...)
is the check's own within-check verdict; `claim_type` is the orthogonal,
epistemic classification of *what kind of claim* that verdict is making.
`grading.py`'s letter aggregation reads `claim_type` and nothing else to decide
what counts -- see its CI guard test for why that separation matters.
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


class ClaimType(str, Enum):
    """What kind of claim a (non-PASS) finding is making -- GRADING_TAXONOMY_V1.md
    section 3. Only DEFECT ever touches the letter grade.

    DEFECT          Objective violation surviving all gates. Touches the letter.
    RISK            A proxy signal within its stated validity domain, proxy
                    named. Reported in the "training_value" profile, never the
                    letter -- it's a training-quality signal, not a correctness
                    claim.
    CHARACTERISTIC  A neutral property (e.g. action/state spaces having
                    different dimensionality by architecture). Index metadata,
                    never a grade input.
    BY_DESIGN       A pre-dated author declaration matching observation, which
                    suppresses what would otherwise be a finding. Reserved --
                    no check in this codebase implements dated-declaration
                    suppression yet, but the type exists so a future one has
                    somewhere correct to report into.
    NOT_ASSESSED    A required gate didn't hold (insufficient data, an
                    undeclared precondition, ...) and the failed gate is named
                    in the summary. Never touches the letter.
    """

    DEFECT = "defect"
    RISK = "risk"
    CHARACTERISTIC = "characteristic"
    BY_DESIGN = "by_design"
    NOT_ASSESSED = "not_assessed"


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
    claim_type: ClaimType
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
