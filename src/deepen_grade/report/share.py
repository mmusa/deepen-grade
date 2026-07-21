# SPDX-License-Identifier: Apache-2.0
"""`--share-report`: an explicit, opt-in path that would POST the JSON report
to Deepen's deepen-robograde.pages.dev for the sealed/signed version of this report.

deepen-grade is privacy-first: nothing uploads by default, full stop. This
module intentionally does NOT perform a network call yet -- it documents the
exact endpoint contract as a TODO so the opt-in behavior is auditable before
it exists, rather than shipping a network call nobody reviewed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# TODO(mmusa/deepen-grade#2): implement the real POST once deepen-robograde.pages.dev
# exposes this endpoint. Contract: POST the exact JSON produced by
# `deepen_grade.report.json_report.build_report`, `Content-Type: application/json`,
# response is a signed report URL. No API key is required for the free tier.
SHARE_ENDPOINT = "https://deepen-robograde.pages.dev/api/v1/reports"


class ShareNotImplementedError(RuntimeError):
    """Raised by every `--share-report` invocation in this release."""


def share_report(report: dict[str, Any], dry_run_path: str | Path | None = None) -> None:
    if dry_run_path is not None:
        Path(dry_run_path).write_text(json.dumps(report, indent=2))
        extra = f" The payload that would be sent was written to {dry_run_path} for inspection."
    else:
        extra = ""
    raise ShareNotImplementedError(
        "--share-report is not implemented in this release: deepen-grade does not "
        f"upload anything yet. It would POST this report to {SHARE_ENDPOINT}."
        + extra
        + " Track/follow along: https://github.com/mmusa/deepen-grade/issues"
    )
