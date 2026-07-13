# SPDX-License-Identifier: Apache-2.0
import numpy as np

from deepen_grade.checks import hygiene
from deepen_grade.checks.base import Severity
from deepen_grade.ingest.base import Dataset, Episode, TfEdge, TopicInfo, Trajectory


def _topic(name, timestamps, msg_type="x"):
    return TopicInfo(name=name, msg_type=msg_type, count=len(timestamps), timestamps_s=np.asarray(timestamps))


def test_topic_frequency_no_drops():
    ts = np.arange(0, 10, 0.1)
    ep = Episode(episode_id="e1", topics=[_topic("/camera/image", ts)])
    result = hygiene.topic_frequency_and_drops(ep)
    assert result.severity == Severity.PASS
    assert result.details["total_gaps"] == 0


def test_topic_frequency_detects_drop():
    ts = list(np.arange(0, 5, 0.1)) + list(np.arange(5.5, 10, 0.1))  # ~0.5s gap
    ep = Episode(episode_id="e1", topics=[_topic("/camera/image", ts)])
    result = hygiene.topic_frequency_and_drops(ep)
    assert result.details["total_gaps"] >= 1
    assert result.severity in (Severity.WARN, Severity.FAIL)


def test_topic_frequency_not_applicable_for_sparse_topic():
    ep = Episode(episode_id="e1", topics=[_topic("/rare", [0.0, 1.0])])  # below MIN_MESSAGES_FOR_FREQ_ANALYSIS
    result = hygiene.topic_frequency_and_drops(ep)
    assert result.severity == Severity.NOT_APPLICABLE


def test_cross_modal_skew_in_sync():
    ts = np.arange(0, 10, 0.05)
    cam = _topic("/camera/image", ts)
    joints = _topic("/joint_states", ts)  # identical clock -> zero skew
    ep = Episode(episode_id="e1", topics=[cam, joints])
    result = hygiene.cross_modal_timestamp_skew(ep)
    assert result.severity == Severity.PASS
    assert all(v <= 1.0 for v in result.details["skew_ms"].values())


def test_cross_modal_skew_detects_offset():
    ts_a = np.arange(0, 10, 0.02)  # high rate reference
    ts_b = ts_a[::4] + 0.08  # 80ms offset, lower rate
    cam = _topic("/camera/image", ts_a)
    joints = _topic("/joint_states", ts_b)
    ep = Episode(episode_id="e1", topics=[cam, joints])
    result = hygiene.cross_modal_timestamp_skew(ep)
    assert result.severity == Severity.FAIL
    assert result.details["skew_ms"]["/joint_states"] > 50.0


def test_tf_tree_na_when_no_tf_concept():
    ep = Episode(episode_id="e1", tf_edges=None)
    result = hygiene.tf_tree_integrity(ep)
    assert result.severity == Severity.NOT_APPLICABLE


def test_tf_tree_valid_single_parent():
    edges = [
        TfEdge("base_link", "arm_link", is_static=False),
        TfEdge("arm_link", "gripper_link", is_static=False),
        TfEdge("base_link", "camera_link", is_static=True),
    ]
    ep = Episode(episode_id="e1", tf_edges=edges)
    result = hygiene.tf_tree_integrity(ep)
    assert result.severity == Severity.PASS
    assert result.details["n_frames"] == 3


def test_tf_tree_detects_multi_parent():
    edges = [
        TfEdge("base_link", "arm_link", is_static=False),
        TfEdge("odom", "arm_link", is_static=False),  # arm_link now has 2 parents
    ]
    ep = Episode(episode_id="e1", tf_edges=edges)
    result = hygiene.tf_tree_integrity(ep)
    assert result.severity == Severity.FAIL
    assert "arm_link" in result.details["multi_parent_frames"]


def test_tf_tree_detects_static_dynamic_conflict():
    edges = [
        TfEdge("base_link", "camera_link", is_static=True),
        TfEdge("base_link", "camera_link", is_static=False),
    ]
    ep = Episode(episode_id="e1", tf_edges=edges)
    result = hygiene.tf_tree_integrity(ep)
    assert result.severity == Severity.WARN


def _episode_with_dim(episode_id, dim, msg_type="sensor_msgs/JointState"):
    traj = Trajectory(
        timestamps_s=np.arange(0, 1, 0.1),
        state=np.zeros((10, dim)),
        state_labels=None,
        action=None,
        action_labels=None,
        gripper_position=None,
    )
    return Episode(
        episode_id=episode_id,
        topics=[_topic("/joint_states", np.arange(0, 1, 0.1), msg_type=msg_type)],
        trajectory=traj,
    )


def test_schema_dim_consistency_pass():
    ds = Dataset(source="x", format="mcap", episodes=[_episode_with_dim("a", 7), _episode_with_dim("b", 7)])
    result = hygiene.schema_and_dim_consistency(ds)
    assert result.severity == Severity.PASS


def test_schema_dim_consistency_detects_dim_mismatch():
    ds = Dataset(source="x", format="mcap", episodes=[_episode_with_dim("a", 7), _episode_with_dim("b", 6)])
    result = hygiene.schema_and_dim_consistency(ds)
    assert result.severity == Severity.FAIL
    assert "b" in result.details["dim_mismatches"]


def test_schema_dim_consistency_detects_schema_mismatch():
    ep_a = _episode_with_dim("a", 7, msg_type="sensor_msgs/JointState")
    ep_b = _episode_with_dim("b", 7, msg_type="sensor_msgs/msg/JointState")
    ds = Dataset(source="x", format="mcap", episodes=[ep_a, ep_b])
    result = hygiene.schema_and_dim_consistency(ds)
    assert result.severity == Severity.FAIL
    assert "/joint_states" in result.details["schema_mismatches"]


def test_schema_dim_consistency_na_for_single_episode():
    ds = Dataset(source="x", format="mcap", episodes=[_episode_with_dim("a", 7)])
    result = hygiene.schema_and_dim_consistency(ds)
    assert result.severity == Severity.NOT_APPLICABLE
