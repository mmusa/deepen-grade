# SPDX-License-Identifier: Apache-2.0
"""Enforces deepen-grade's core promise: every check that reports something
other than N/A must cite a real, public source."""

import pytest

from deepen_grade.checks.base import CheckResult, ClaimType, Severity
from deepen_grade.citations import ALL_CITATIONS


def test_all_citations_have_required_fields():
    for key, c in ALL_CITATIONS.items():
        assert c.key == key
        assert c.title and c.authors and c.venue and c.url
        assert c.url.startswith("http")


def test_check_result_without_citation_raises_unless_na():
    with pytest.raises(ValueError):
        CheckResult(check_id="x", name="x", severity=Severity.PASS, summary="s", citation_keys=(),
                    claim_type=ClaimType.DEFECT)


def test_check_result_na_may_skip_citation():
    result = CheckResult(check_id="x", name="x", severity=Severity.NOT_APPLICABLE, summary="s", citation_keys=(),
                          claim_type=ClaimType.NOT_ASSESSED)
    assert result.severity == Severity.NOT_APPLICABLE


def test_check_result_citation_keys_resolve():
    result = CheckResult(check_id="x", name="x", severity=Severity.WARN, summary="s", citation_keys=("dqaf",),
                          claim_type=ClaimType.RISK)
    for key in result.citation_keys:
        assert key in ALL_CITATIONS
