# SPDX-License-Identifier: Apache-2.0
"""End-to-end ingestion tests against the real generated fixtures (see
tests/fixtures/generate.py) -- these exercise the actual mcap/rosbags/parquet
libraries, not mocks."""

import pytest

from deepen_grade.grading import grade_dataset
from deepen_grade.ingest.detect import detect_format, load_dataset


def test_detect_format_mcap(sample_mcap):
    assert detect_format(str(sample_mcap)) == "mcap"


def test_detect_format_ros1(sample_ros1_bag):
    assert detect_format(str(sample_ros1_bag)) == "ros1_bag"


def test_detect_format_ros2(sample_ros2_bag):
    assert detect_format(str(sample_ros2_bag)) == "ros2_db3"


def test_detect_format_lerobot(sample_lerobot):
    assert detect_format(str(sample_lerobot)) == "lerobot"


def test_detect_format_repo_id_heuristic():
    assert detect_format("some-org/some-dataset") == "lerobot"


def test_detect_format_unknown_extension(tmp_path):
    from deepen_grade.ingest.exceptions import UnsupportedFormatError

    bad = tmp_path / "file.xyz"
    bad.write_text("nope")
    with pytest.raises(UnsupportedFormatError):
        detect_format(str(bad))


@pytest.mark.parametrize("fixture_name", ["sample_mcap", "sample_ros1_bag", "sample_ros2_bag", "sample_lerobot"])
def test_load_and_grade_each_format(fixture_name, request):
    path = request.getfixturevalue(fixture_name)
    dataset = load_dataset(str(path))
    assert dataset.episodes, f"{fixture_name}: no episodes loaded"
    for ep in dataset.episodes:
        assert ep.trajectory is not None, f"{fixture_name}/{ep.episode_id}: expected decodable trajectory"
        assert ep.trajectory.state is not None
        assert ep.trajectory.gripper_position is not None

    grade = grade_dataset(dataset)
    assert 0 <= grade.overall_score <= 100
    assert grade.overall_letter in ("A", "B", "C", "D", "F")
    # every episode ran all three check families (3 hygiene + 5 quality + 1 calibration = 9)
    for eg in grade.episode_grades:
        assert len(eg.results) == 9


def test_mcap_gripper_chatter_burst_detected(sample_mcap):
    dataset = load_dataset(str(sample_mcap))
    grade = grade_dataset(dataset)
    chatter = next(r for r in grade.episode_grades[0].results if r.check_id == "episode_quality.gripper_chatter")
    assert chatter.severity.value == "fail"
    assert chatter.details["peak_rate_hz"] >= 5.0


def test_mcap_cross_modal_skew_detected(sample_mcap):
    dataset = load_dataset(str(sample_mcap))
    grade = grade_dataset(dataset)
    skew = next(r for r in grade.episode_grades[0].results if r.check_id == "hygiene.cross_modal_sync")
    assert skew.severity.value == "warn"
    assert skew.details["skew_ms"]["/camera/image_raw"] > 15.0


def test_lerobot_gripper_label_routing(sample_lerobot):
    dataset = load_dataset(str(sample_lerobot))
    ep = dataset.episodes[0]
    assert ep.trajectory.state.shape[1] == 4  # 3 joints + gripper
    assert "gripper" in ep.trajectory.state_labels
    assert ep.trajectory.gripper_position is not None


def test_missing_optional_dependency_gives_actionable_error(monkeypatch, sample_mcap):
    from deepen_grade.ingest import mcap_reader

    def _boom():
        from deepen_grade.ingest.exceptions import MissingOptionalDependencyError

        raise MissingOptionalDependencyError(feature=".mcap files", extra="mcap", packages=["mcap"])

    monkeypatch.setattr(mcap_reader, "_require_mcap", _boom)
    from deepen_grade.ingest.exceptions import MissingOptionalDependencyError

    with pytest.raises(MissingOptionalDependencyError, match="deepen-grade\\[mcap\\]"):
        mcap_reader.read_mcap(str(sample_mcap))
