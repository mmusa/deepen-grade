# SPDX-License-Identifier: Apache-2.0
"""Errors that ingestion adapters raise. Kept small and user-facing on purpose --
`deepen grade` prints these directly to the terminal, so the message is the fix.
"""

from __future__ import annotations


class MissingOptionalDependencyError(RuntimeError):
    """Raised when a format's reader is invoked but its extra wasn't installed."""

    def __init__(self, feature: str, extra: str, packages: list[str]) -> None:
        pkgs = ", ".join(packages)
        message = (
            f"Reading {feature} requires the optional '{extra}' extra "
            f"({pkgs}), which isn't installed.\n\n"
            f'    pip install "deepen-grade[{extra}] @ git+https://github.com/mmusa/deepen-grade"\n'
        )
        super().__init__(message)
        self.feature = feature
        self.extra = extra


class UnsupportedFormatError(RuntimeError):
    """Raised when the input path/repo-id can't be matched to a known format."""


class DatasetReadError(RuntimeError):
    """Raised when a file matches a known format but can't be parsed."""
