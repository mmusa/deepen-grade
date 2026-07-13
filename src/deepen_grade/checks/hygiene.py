# SPDX-License-Identifier: Apache-2.0
"""Hygiene checks: deterministic, zero-research-risk structural checks that
apply to any recording regardless of robot or task -- topic frequency/drops,
cross-modal timestamp skew, tf-tree integrity, and message-schema / state-dim
consistency across a dataset's episodes.

Grounded in REP 105 (ROS coordinate-frame conventions) and the ISO/WD 26264-1
companion paper's explicit requirement that "timing, coordinate frames,
calibration, kinematics, units, and synchronization assumptions remain
inspectable" (arXiv:2606.19769) -- these checks are exactly that inspection.
"""

from __future__ import annotations

import numpy as np

from deepen_grade.checks.base import CheckResult, Severity
from deepen_grade.citations import ASAM_OPENLABEL, DATA_STANDARDS_HUMANOID, REP_105
from deepen_grade.ingest.base import Dataset, Episode

# A gap this many times a topic's own median inter-message interval counts as
# a drop. Simple, transparent, and documented -- not a tuned/sealed threshold.
DROP_GAP_MULTIPLIER = 3.0
MIN_MESSAGES_FOR_FREQ_ANALYSIS = 3

# Cross-modal sync skew bands (ms). A camera/IMU/lidar system that's out of
# sync by more than a frame period at typical robot-learning capture rates
# (20-60 Hz => ~15-50ms/frame) is the generic "you probably have a sync bug"
# threshold; datasets built on truly frame-synced formats (LeRobot) never hit
# this because every feature shares one recorded timestamp.
SKEW_WARN_MS = 15.0
SKEW_FAIL_MS = 50.0


def topic_frequency_and_drops(episode: Episode) -> CheckResult:
    per_topic: dict[str, dict] = {}
    total_gaps = 0
    worst_topic = None

    for topic in episode.topics:
        if len(topic.timestamps_s) < MIN_MESSAGES_FOR_FREQ_ANALYSIS:
            continue
        dt = np.diff(topic.timestamps_s)
        median_dt = float(np.median(dt))
        if median_dt <= 0:
            continue
        n_gaps = int(np.sum(dt > DROP_GAP_MULTIPLIER * median_dt))
        per_topic[topic.name] = {
            "observed_hz": round(1.0 / median_dt, 2),
            "count": topic.count,
            "n_gaps": n_gaps,
            "max_gap_s": round(float(dt.max()), 3),
        }
        total_gaps += n_gaps
        if worst_topic is None or n_gaps > per_topic[worst_topic]["n_gaps"]:
            worst_topic = topic.name

    if not per_topic:
        return CheckResult(
            check_id="hygiene.topic_frequency",
            name="Topic frequency & drop analysis",
            severity=Severity.NOT_APPLICABLE,
            summary="not enough messages per topic to analyze frequency",
            citation_keys=(),
        )

    if total_gaps == 0:
        severity, summary = Severity.PASS, f"{len(per_topic)} topics, no drops detected"
    elif total_gaps <= 2:
        severity = Severity.WARN
        summary = f"{total_gaps} gap(s) across {len(per_topic)} topics; worst: {worst_topic}"
    else:
        severity = Severity.FAIL
        summary = (
            f"{total_gaps} gaps across {len(per_topic)} topics; worst: {worst_topic} "
            f"({per_topic[worst_topic]['n_gaps']} gaps)"
        )

    return CheckResult(
        check_id="hygiene.topic_frequency",
        name="Topic frequency & drop analysis",
        severity=severity,
        summary=summary,
        citation_keys=(DATA_STANDARDS_HUMANOID.key,),
        details={"topics": per_topic, "total_gaps": total_gaps,
                 "drop_gap_multiplier": DROP_GAP_MULTIPLIER},
    )


def cross_modal_timestamp_skew(episode: Episode) -> CheckResult:
    """How much later/earlier does each modality's stream start/end relative
    to the others? A stream that starts late (driver came up after the
    others) or ends early (dropped mid-episode) is a real, common sync bug.

    Deliberately anchor-based (first/last message time), not nearest-neighbor
    matching across the whole series: a densely-sampled reference stream
    would otherwise always have *some* sample close in time to any other
    stream's timestamp, masking a genuine constant clock offset between the
    two streams entirely.
    """
    modal_topics = [
        t for t in episode.topics
        if t.modality in ("vision", "proprioception", "lidar") and len(t.timestamps_s) >= 1
    ]
    if len(modal_topics) < 2:
        return CheckResult(
            check_id="hygiene.cross_modal_sync",
            name="Cross-modal timestamp skew",
            severity=Severity.NOT_APPLICABLE,
            summary="fewer than 2 timestamped sensing modalities present",
            citation_keys=(),
        )

    starts = {t.name: float(t.timestamps_s[0]) for t in modal_topics}
    ends = {t.name: float(t.timestamps_s[-1]) for t in modal_topics}
    earliest_start, latest_end = min(starts.values()), max(ends.values())
    skews_ms = {
        name: round(max(starts[name] - earliest_start, latest_end - ends[name]) * 1000.0, 2)
        for name in starts
    }
    max_skew = max(skews_ms.values())
    worst_topic = max(skews_ms, key=skews_ms.get)

    if max_skew <= SKEW_WARN_MS:
        severity, summary = Severity.PASS, f"max cross-modal skew {max_skew:.1f}ms"
    elif max_skew <= SKEW_FAIL_MS:
        severity = Severity.WARN
        summary = f"max cross-modal skew {max_skew:.1f}ms ({worst_topic}) exceeds {SKEW_WARN_MS:.0f}ms budget"
    else:
        severity = Severity.FAIL
        summary = f"max cross-modal skew {max_skew:.1f}ms ({worst_topic}) exceeds {SKEW_FAIL_MS:.0f}ms budget"

    return CheckResult(
        check_id="hygiene.cross_modal_sync",
        name="Cross-modal timestamp skew",
        severity=severity,
        summary=summary,
        citation_keys=(DATA_STANDARDS_HUMANOID.key,),
        details={"skew_ms": skews_ms, "warn_ms": SKEW_WARN_MS, "fail_ms": SKEW_FAIL_MS},
    )


def tf_tree_integrity(episode: Episode) -> CheckResult:
    if episode.tf_edges is None:
        return CheckResult(
            check_id="hygiene.tf_tree_integrity",
            name="tf-tree integrity",
            severity=Severity.NOT_APPLICABLE,
            summary="format has no tf concept",
            citation_keys=(),
        )
    if not episode.tf_edges:
        return CheckResult(
            check_id="hygiene.tf_tree_integrity",
            name="tf-tree integrity",
            severity=Severity.NOT_APPLICABLE,
            summary="no /tf or /tf_static messages found in this recording",
            citation_keys=(),
        )

    parents_of: dict[str, set[str]] = {}
    static_edges: set[tuple[str, str]] = set()
    dynamic_edges: set[tuple[str, str]] = set()
    for edge in episode.tf_edges:
        parents_of.setdefault(edge.child_frame, set()).add(edge.parent_frame)
        (static_edges if edge.is_static else dynamic_edges).add((edge.parent_frame, edge.child_frame))

    multi_parent = {child: sorted(parents) for child, parents in parents_of.items() if len(parents) > 1}
    static_dynamic_conflicts = sorted(static_edges & dynamic_edges)
    cycle = _find_cycle(parents_of)

    if multi_parent or cycle:
        severity = Severity.FAIL
        reason = f"{len(multi_parent)} frame(s) with multiple parents" if multi_parent else f"cycle at {cycle}"
        summary = f"broken tf tree: {reason} (REP 105 requires a single-parent tree)"
    elif static_dynamic_conflicts:
        severity = Severity.WARN
        summary = f"{len(static_dynamic_conflicts)} edge(s) published as both static and dynamic"
    else:
        n_frames = len(parents_of)
        summary = f"{n_frames} frames, single-parent tree, no static/dynamic conflicts"
        severity = Severity.PASS

    return CheckResult(
        check_id="hygiene.tf_tree_integrity",
        name="tf-tree integrity",
        severity=severity,
        summary=summary,
        citation_keys=(REP_105.key,),
        details={
            "n_frames": len(parents_of),
            "multi_parent_frames": multi_parent,
            "static_dynamic_conflicts": [f"{p}->{c}" for p, c in static_dynamic_conflicts],
            "cycle": cycle,
        },
    )


def _find_cycle(parents_of: dict[str, set[str]]) -> list[str] | None:
    """DFS for a cycle in the (child -> parents) graph. Returns the cycle path, or None."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in parents_of}
    path: list[str] = []

    def visit(node: str) -> list[str] | None:
        color[node] = GRAY
        path.append(node)
        for parent in parents_of.get(node, ()):
            if color.get(parent, WHITE) == GRAY:
                return path[path.index(parent):] + [parent]
            if color.get(parent, WHITE) == WHITE and parent in parents_of:
                found = visit(parent)
                if found:
                    return found
        path.pop()
        color[node] = BLACK
        return None

    for node in list(parents_of):
        if color[node] == WHITE:
            found = visit(node)
            if found:
                return found
    return None


def schema_and_dim_consistency(dataset: Dataset) -> CheckResult:
    if len(dataset.episodes) < 2:
        return CheckResult(
            check_id="hygiene.schema_dim_consistency",
            name="Message-schema & action/state-dim consistency",
            severity=Severity.NOT_APPLICABLE,
            summary="only one episode in this dataset; nothing to compare across episodes",
            citation_keys=(),
        )

    type_by_name: dict[str, set[str]] = {}
    for ep in dataset.episodes:
        for topic in ep.topics:
            type_by_name.setdefault(topic.name, set()).add(topic.msg_type or "<unknown>")
    schema_mismatches = {name: sorted(types) for name, types in type_by_name.items() if len(types) > 1}

    dim_by_episode: dict[str, int] = {}
    ref_dim = None
    dim_mismatches: list[str] = []
    for ep in dataset.episodes:
        if ep.trajectory is None or ep.trajectory.state is None:
            continue
        dim = int(ep.trajectory.state.shape[1])
        dim_by_episode[ep.episode_id] = dim
        if ref_dim is None:
            ref_dim = dim
        elif dim != ref_dim:
            dim_mismatches.append(ep.episode_id)

    if schema_mismatches or dim_mismatches:
        severity = Severity.FAIL
        parts = []
        if schema_mismatches:
            parts.append(f"{len(schema_mismatches)} topic(s) with inconsistent message types")
        if dim_mismatches:
            parts.append(f"{len(dim_mismatches)} episode(s) with a state-dim mismatch (expected {ref_dim})")
        summary = "; ".join(parts)
    else:
        summary = f"{len(dataset.episodes)} episodes: consistent schemas and state dims"
        severity = Severity.PASS

    return CheckResult(
        check_id="hygiene.schema_dim_consistency",
        name="Message-schema & action/state-dim consistency",
        severity=severity,
        summary=summary,
        citation_keys=(DATA_STANDARDS_HUMANOID.key, ASAM_OPENLABEL.key),
        details={
            "schema_mismatches": schema_mismatches,
            "dim_mismatches": dim_mismatches,
            "expected_state_dim": ref_dim,
            "dim_by_episode": dim_by_episode,
        },
    )


def run_episode_checks(episode: Episode) -> list[CheckResult]:
    return [
        topic_frequency_and_drops(episode),
        cross_modal_timestamp_skew(episode),
        tf_tree_integrity(episode),
    ]


def run_dataset_checks(dataset: Dataset) -> list[CheckResult]:
    return [schema_and_dim_consistency(dataset)]
