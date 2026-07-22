# SPDX-License-Identifier: Apache-2.0
"""Golden-corpus regression test (grade_schema 1.0.0, GRADING_TAXONOMY_V1.md
section 9's freeze condition): grades every checked-in fixture dataset and
asserts the FULL expected outcome -- overall grade, per-check-id defect
density, and every episode's (severity, claim_type) per check.

This is deliberately exhaustive, not a spot-check: an intentional change to a
check's threshold, claim type, or the aggregation function must force a
conscious edit to the `_EXPECTED` tables below, not silently pass because the
test only looked at one field. If you're here because this test failed after
a real check change, update the expectation and explain why in the same
commit -- that's the golden-corpus discipline, not a bug in the test.
"""

from __future__ import annotations

from deepen_grade.grading import grade_dataset
from deepen_grade.ingest.detect import load_dataset

# Per-fixture: (overall score, overall letter), {check_id: density} for every
# DEFECT-class check that fired, and the sorted list of RISK check_ids that
# appear in training_value.
_EXPECTED_OVERALL = {
    "sample.mcap": (100, "A"),
    "sample_ros1.bag": (100, "A"),
    "sample_ros2": (100, "A"),
    "sample_lerobot": (100, "A"),
}

_EXPECTED_DEFECT_CLASSES = {
    "sample.mcap": {"hygiene.topic_frequency": 0.0, "hygiene.tf_tree_integrity": 0.0},
    "sample_ros1.bag": {"hygiene.topic_frequency": 0.0},
    "sample_ros2": {"hygiene.topic_frequency": 0.0},
    "sample_lerobot": {"hygiene.schema_dim_consistency": 0.0, "hygiene.topic_frequency": 0.0},
}

_EXPECTED_TRAINING_VALUE_CHECKS = {
    "sample.mcap": {
        "episode_quality.gripper_chatter", "episode_quality.idle_stall",
        "episode_quality.jerk_smoothness", "episode_quality.joint_limit_saturation",
        "hygiene.cross_modal_sync",
    },
    "sample_ros1.bag": {
        "episode_quality.gripper_chatter", "episode_quality.idle_stall",
        "episode_quality.jerk_smoothness", "episode_quality.joint_limit_saturation",
        "hygiene.cross_modal_sync",
    },
    "sample_ros2": {
        "episode_quality.gripper_chatter", "episode_quality.idle_stall",
        "episode_quality.jerk_smoothness", "episode_quality.joint_limit_saturation",
        "hygiene.cross_modal_sync",
    },
    "sample_lerobot": {
        "episode_quality.gripper_chatter", "episode_quality.idle_stall",
        "episode_quality.jerk_smoothness", "episode_quality.joint_limit_saturation",
    },
}

# Per-fixture, per-episode: {check_id: (severity, claim_type)}. mcap/ros1/ros2
# fixtures have one episode ("sample"/"sample_ros1"/"sample_ros2"); lerobot has two.
_EXPECTED_CHECKS = {
    "sample.mcap": {
        "sample": {
            "hygiene.topic_frequency": ("pass", "defect"),
            "hygiene.cross_modal_sync": ("warn", "risk"),
            "hygiene.tf_tree_integrity": ("pass", "defect"),
            "episode_quality.idle_stall": ("pass", "risk"),
            "episode_quality.jerk_smoothness": ("pass", "risk"),
            "episode_quality.joint_limit_saturation": ("warn", "risk"),
            "episode_quality.gripper_chatter": ("fail", "risk"),
            "episode_quality.action_state_consistency": ("n/a", "not_assessed"),
            "calibration.sanity": ("pass", "defect"),
        },
    },
    "sample_ros1.bag": {
        "sample_ros1": {
            "hygiene.topic_frequency": ("pass", "defect"),
            "hygiene.cross_modal_sync": ("warn", "risk"),
            "hygiene.tf_tree_integrity": ("n/a", "not_assessed"),
            "episode_quality.idle_stall": ("pass", "risk"),
            "episode_quality.jerk_smoothness": ("pass", "risk"),
            "episode_quality.joint_limit_saturation": ("warn", "risk"),
            "episode_quality.gripper_chatter": ("fail", "risk"),
            "episode_quality.action_state_consistency": ("n/a", "not_assessed"),
            "calibration.sanity": ("info", "not_assessed"),
        },
    },
    "sample_ros2": {
        "sample_ros2": {
            "hygiene.topic_frequency": ("pass", "defect"),
            "hygiene.cross_modal_sync": ("pass", "risk"),
            "hygiene.tf_tree_integrity": ("n/a", "not_assessed"),
            "episode_quality.idle_stall": ("pass", "risk"),
            "episode_quality.jerk_smoothness": ("pass", "risk"),
            "episode_quality.joint_limit_saturation": ("warn", "risk"),
            "episode_quality.gripper_chatter": ("fail", "risk"),
            "episode_quality.action_state_consistency": ("n/a", "not_assessed"),
            "calibration.sanity": ("info", "not_assessed"),
        },
    },
    "sample_lerobot": {
        "episode_000000": {
            "hygiene.topic_frequency": ("pass", "defect"),
            "hygiene.cross_modal_sync": ("n/a", "not_assessed"),
            "hygiene.tf_tree_integrity": ("n/a", "not_assessed"),
            "episode_quality.idle_stall": ("pass", "risk"),
            "episode_quality.jerk_smoothness": ("pass", "risk"),
            "episode_quality.joint_limit_saturation": ("fail", "risk"),
            "episode_quality.gripper_chatter": ("fail", "risk"),
            "episode_quality.action_state_consistency": ("n/a", "not_assessed"),
            "calibration.sanity": ("info", "not_assessed"),
        },
        "episode_000001": {
            "hygiene.topic_frequency": ("pass", "defect"),
            "hygiene.cross_modal_sync": ("n/a", "not_assessed"),
            "hygiene.tf_tree_integrity": ("n/a", "not_assessed"),
            "episode_quality.idle_stall": ("pass", "risk"),
            "episode_quality.jerk_smoothness": ("pass", "risk"),
            "episode_quality.joint_limit_saturation": ("fail", "risk"),
            "episode_quality.gripper_chatter": ("pass", "risk"),
            "episode_quality.action_state_consistency": ("n/a", "not_assessed"),
            "calibration.sanity": ("info", "not_assessed"),
        },
    },
}


def _grade(fixture_name, request):
    path = request.getfixturevalue(fixture_name)
    dataset = load_dataset(str(path))
    return grade_dataset(dataset)


def test_golden_overall_grade(sample_mcap, sample_ros1_bag, sample_ros2_bag, sample_lerobot):
    fixtures = {
        "sample.mcap": sample_mcap, "sample_ros1.bag": sample_ros1_bag,
        "sample_ros2": sample_ros2_bag, "sample_lerobot": sample_lerobot,
    }
    for name, path in fixtures.items():
        grade = grade_dataset(load_dataset(str(path)))
        expected_score, expected_letter = _EXPECTED_OVERALL[name]
        assert (grade.overall_score, grade.overall_letter) == (expected_score, expected_letter), name


def test_golden_defect_classes(sample_mcap, sample_ros1_bag, sample_ros2_bag, sample_lerobot):
    fixtures = {
        "sample.mcap": sample_mcap, "sample_ros1.bag": sample_ros1_bag,
        "sample_ros2": sample_ros2_bag, "sample_lerobot": sample_lerobot,
    }
    for name, path in fixtures.items():
        grade = grade_dataset(load_dataset(str(path)))
        densities = {check_id: entry["density"] for check_id, entry in grade.defect_classes.items()}
        assert densities == _EXPECTED_DEFECT_CLASSES[name], name


def test_golden_training_value_checks(sample_mcap, sample_ros1_bag, sample_ros2_bag, sample_lerobot):
    fixtures = {
        "sample.mcap": sample_mcap, "sample_ros1.bag": sample_ros1_bag,
        "sample_ros2": sample_ros2_bag, "sample_lerobot": sample_lerobot,
    }
    for name, path in fixtures.items():
        grade = grade_dataset(load_dataset(str(path)))
        assert set(grade.training_value.keys()) == _EXPECTED_TRAINING_VALUE_CHECKS[name], name


def test_golden_per_episode_checks(sample_mcap, sample_ros1_bag, sample_ros2_bag, sample_lerobot):
    fixtures = {
        "sample.mcap": sample_mcap, "sample_ros1.bag": sample_ros1_bag,
        "sample_ros2": sample_ros2_bag, "sample_lerobot": sample_lerobot,
    }
    for name, path in fixtures.items():
        grade = grade_dataset(load_dataset(str(path)))
        expected_episodes = _EXPECTED_CHECKS[name]
        actual_ids = {eg.episode_id for eg in grade.episode_grades}
        assert actual_ids == set(expected_episodes), name
        for eg in grade.episode_grades:
            actual = {r.check_id: (r.severity.value, r.claim_type.value) for r in eg.results}
            assert actual == expected_episodes[eg.episode_id], f"{name}/{eg.episode_id}"
