# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

FIXTURES_DATA = Path(__file__).parent / "fixtures" / "data"


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"fixture {path} not generated -- run `python tests/fixtures/generate.py`")
    return path


@pytest.fixture
def sample_mcap() -> Path:
    return _require(FIXTURES_DATA / "sample.mcap")


@pytest.fixture
def sample_ros1_bag() -> Path:
    return _require(FIXTURES_DATA / "sample_ros1.bag")


@pytest.fixture
def sample_ros2_bag() -> Path:
    return _require(FIXTURES_DATA / "sample_ros2")


@pytest.fixture
def sample_lerobot() -> Path:
    return _require(FIXTURES_DATA / "sample_lerobot")
