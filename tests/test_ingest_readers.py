# SPDX-License-Identifier: Apache-2.0
"""End-to-end ingestion tests against the real generated fixtures (see
tests/fixtures/generate.py) -- these exercise the actual mcap/rosbags/parquet
libraries, not mocks."""

import sys
from pathlib import Path

import pytest

from deepen_grade.grading import grade_dataset
from deepen_grade.ingest.detect import detect_format, load_dataset

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from generate import CAMERA_EXTRINSICS, CAMERA_FRAME_ID, CAMERA_K, write_lerobot  # noqa: E402


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


def test_mcap_camera_info_and_tf_static_extraction(sample_mcap):
    """Feature 7: CameraInfo intrinsics paired with /tf_static extrinsics by
    frame_id -- 0/30 real datasets sampled carried this, so the fixture is
    the only place this path gets exercised end-to-end."""
    dataset = load_dataset(str(sample_mcap))
    calibrations = dataset.episodes[0].calibrations
    assert len(calibrations) == 1
    cal = calibrations[0]
    assert cal.camera_id == CAMERA_FRAME_ID
    assert cal.source == "camera_info"
    assert cal.intrinsics["fx"] == CAMERA_K[0]
    assert cal.intrinsics["fy"] == CAMERA_K[4]
    assert cal.intrinsics["cx"] == CAMERA_K[2]
    assert cal.intrinsics["cy"] == CAMERA_K[5]
    assert cal.extrinsics == CAMERA_EXTRINSICS


def test_mcap_camera_info_does_not_pollute_cross_modal_sync(sample_mcap):
    """The CameraInfo/tf_static calibration carriers are one-shot metadata,
    not sensing streams -- they must not feed the vision-modality cross-modal
    sync-skew check alongside the real /camera/image_raw stream."""
    dataset = load_dataset(str(sample_mcap))
    grade = grade_dataset(dataset)
    skew = next(r for r in grade.episode_grades[0].results if r.check_id == "hygiene.cross_modal_sync")
    assert skew.severity.value == "warn"  # unchanged from the pre-calibration-carrier fixture


@pytest.fixture
def nested_lerobot_container(tmp_path) -> Path:
    """A container repo: two independent, full LeRobot datasets nested one
    directory down under a parent with no meta/info.json of its own -- the
    real-world shape hit in 11/30 of the top-downloaded LeRobot repos on the
    Hub (e.g. a per-contributor directory layout)."""
    root = tmp_path / "container_repo"
    write_lerobot(root / "contributor_a" / "session1")
    write_lerobot(root / "contributor_b" / "session2")
    return root


def test_detect_format_nested_container(nested_lerobot_container):
    assert detect_format(str(nested_lerobot_container)) == "lerobot"


def test_container_repo_loads_all_subdatasets_with_prefixed_ids(nested_lerobot_container):
    dataset = load_dataset(str(nested_lerobot_container))
    assert len(dataset.episodes) == 4  # 2 episodes x 2 subdatasets
    ids = {ep.episode_id for ep in dataset.episodes}
    assert ids == {
        "contributor_a/session1::episode_000000", "contributor_a/session1::episode_000001",
        "contributor_b/session2::episode_000000", "contributor_b/session2::episode_000001",
    }
    assert any("container repo" in w and "2 nested" in w for w in dataset.warnings)


def test_subdataset_flag_grades_just_one_nested_dataset(nested_lerobot_container):
    dataset = load_dataset(str(nested_lerobot_container), subdataset="contributor_a/session1")
    assert len(dataset.episodes) == 2
    assert dataset.episodes[0].episode_id == "episode_000000"  # unprefixed: same as a plain single dataset
    assert not any("container repo" in w for w in dataset.warnings)


def test_subdataset_flag_unknown_path_raises_actionable_error(nested_lerobot_container):
    from deepen_grade.ingest.exceptions import DatasetReadError

    with pytest.raises(DatasetReadError, match="not found under"):
        load_dataset(str(nested_lerobot_container), subdataset="does/not/exist")


def test_single_dataset_behavior_is_unaffected_by_container_support(sample_lerobot):
    """A plain (non-container) LeRobot dataset must load byte-identically to
    before: same episode IDs, no container warning."""
    dataset = load_dataset(str(sample_lerobot))
    assert [ep.episode_id for ep in dataset.episodes] == ["episode_000000", "episode_000001"]
    assert not any("container repo" in w for w in dataset.warnings)


def test_sample_is_deterministic_for_a_given_seed(sample_lerobot):
    ds_a = load_dataset(str(sample_lerobot), sample=1, seed=7)
    ds_b = load_dataset(str(sample_lerobot), sample=1, seed=7)
    assert [ep.episode_id for ep in ds_a.episodes] == [ep.episode_id for ep in ds_b.episodes]
    assert ds_a.sampling == {"mode": "sample", "n": 1, "seed": 7, "episodes_total": 2}


def test_sample_different_seeds_can_pick_different_episodes(sample_lerobot):
    # sample_lerobot has exactly 2 episodes; try enough seeds that at least
    # one must differ from seed 0's pick if selection is seed-sensitive at all.
    picks = {load_dataset(str(sample_lerobot), sample=1, seed=s).episodes[0].episode_id for s in range(10)}
    assert len(picks) == 2


def test_max_episodes_takes_first_n_in_natural_order(sample_lerobot):
    dataset = load_dataset(str(sample_lerobot), max_episodes=1)
    assert [ep.episode_id for ep in dataset.episodes] == ["episode_000000"]
    assert dataset.sampling == {"mode": "head", "n": 1, "seed": None, "episodes_total": 2}


def test_no_sampling_flags_leaves_sampling_none(sample_lerobot):
    dataset = load_dataset(str(sample_lerobot))
    assert dataset.sampling is None


def test_droid_style_cam2base_extrinsics_json_is_read(tmp_path):
    """Narrowly-scoped DROID-style calibration convention (feature 7): a
    meta/cam2base_extrinsics.json mapping camera_id -> a base-frame extrinsic
    vector. No intrinsics claim is made from this file, so calibration_sanity
    correctly flags "intrinsics missing" -- that's an accurate signal, not a bug."""
    import json

    root = tmp_path / "droid_style"
    write_lerobot(root)
    (root / "meta" / "cam2base_extrinsics.json").write_text(
        json.dumps({"cam_0": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]})
    )
    dataset = load_dataset(str(root))
    calibrations = dataset.episodes[0].calibrations
    assert len(calibrations) == 1
    assert calibrations[0].camera_id == "cam_0"
    assert calibrations[0].extrinsics == [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    assert calibrations[0].intrinsics is None
    assert calibrations[0].source == "meta/cam2base_extrinsics.json"


# --- QA defect 1: nested feature "names" shape (real container repo) --------


def test_nested_state_names_does_not_crash_and_flattens(tmp_path):
    """Real-world regression: JeffrinSam/G1-BrainCo-Pick-GR00T stores
    features["observation.state"]["names"] as a single-element outer list
    wrapping the actual 26-name joint list ([[j0, ..., j25]]), not a flat
    list[str] as documented. Reading this used to raise AttributeError deep
    in _extract_gripper (name.lower() on a list). It must instead flatten the
    one level of nesting and keep grading normally."""
    import json

    root = tmp_path / "nested_names"
    write_lerobot(root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    flat_names = info["features"]["observation.state"]["names"]
    info["features"]["observation.state"]["names"] = [flat_names]  # copy the real nested shape
    info_path.write_text(json.dumps(info))

    dataset = load_dataset(str(root))  # must not raise
    ep = dataset.episodes[0]
    assert ep.trajectory.state_labels == flat_names  # flattened back to the original flat list
    assert ep.trajectory.gripper_position is not None  # gripper routing still works post-flatten


def test_doubly_nested_or_malformed_names_degrades_to_none_not_crash(tmp_path):
    """Anything stranger than one level of nesting (e.g. multiple groups) is
    ambiguous to unwrap -- must degrade to labels=None, never crash."""
    import json

    root = tmp_path / "malformed_names"
    write_lerobot(root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.state"]["names"] = [["a", "b"], ["c", "d"]]  # multi-group: ambiguous
    info_path.write_text(json.dumps(info))

    dataset = load_dataset(str(root))  # must not raise
    assert dataset.episodes[0].trajectory.state_labels is None


# --- QA defect 2: container fault isolation ----------------------------------


def _corrupt_episode_index_column(sub_root: Path) -> None:
    import pandas as pd

    for f in (sub_root / "data" / "chunk-000").glob("*.parquet"):
        pd.read_parquet(f).drop(columns=["episode_index"]).to_parquet(f)


def test_one_broken_subdataset_does_not_sink_the_whole_container(tmp_path):
    root = tmp_path / "mixed_container"
    write_lerobot(root / "good" / "session1")
    write_lerobot(root / "broken" / "session2")
    _corrupt_episode_index_column(root / "broken" / "session2")

    dataset = load_dataset(str(root))  # must not raise
    assert [ep.episode_id for ep in dataset.episodes] == [
        "good/session1::episode_000000", "good/session1::episode_000001",
    ]
    assert any(
        w.startswith("sub-dataset broken/session2 could not be read:") for w in dataset.warnings
    )


def test_all_subdatasets_broken_raises_cleanly(tmp_path):
    from deepen_grade.ingest.exceptions import DatasetReadError

    root = tmp_path / "all_broken_container"
    write_lerobot(root / "a" / "session1")
    write_lerobot(root / "b" / "session2")
    _corrupt_episode_index_column(root / "a" / "session1")
    _corrupt_episode_index_column(root / "b" / "session2")

    with pytest.raises(DatasetReadError, match="every nested sub-dataset failed to read"):
        load_dataset(str(root))


def test_single_dataset_read_failure_still_raises_normally(tmp_path):
    """A broken *non-container* dataset (or an explicit --subdataset pick)
    must still fail loudly -- container fault isolation is scoped to grading
    multiple nested sub-datasets together, not to hiding a lone dataset's
    own errors."""
    from deepen_grade.ingest.exceptions import DatasetReadError

    root = tmp_path / "single_broken"
    write_lerobot(root)
    _corrupt_episode_index_column(root)

    with pytest.raises(DatasetReadError, match="no 'episode_index' column"):
        load_dataset(str(root))
