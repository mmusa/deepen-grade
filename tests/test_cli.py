# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from click.testing import CliRunner

from deepen_grade.cli import main

# check_ids allowed to report a non-N/A severity without a citation:
# plausibility's INFO firing is a deliberately uncited content heuristic (see
# checks/plausibility.py's module docstring and the exemption in
# checks/base.py) -- no other check_id belongs here.
UNCITED_CHECK_IDS = {"plausibility.robot_data"}


def test_grade_terminal_output(sample_mcap):
    result = CliRunner().invoke(main, ["grade", str(sample_mcap)])
    assert result.exit_code == 0
    assert "deepen-grade report" in result.output
    assert "Overall grade" in result.output


def test_grade_json_output_is_valid(sample_mcap):
    result = CliRunner().invoke(main, ["grade", str(sample_mcap), "--json"])
    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["tool"] == "deepen-grade"
    assert report["format"] == "mcap"
    assert "overall" in report and "score" in report["overall"]
    assert report["episodes"], "expected at least one episode"
    assert report["partial"] is False
    all_checks = report["dataset_level_checks"] + [c for ep in report["episodes"] for c in ep["checks"]]
    for check in all_checks:
        if check["severity"] != "n/a" and check["check_id"] not in UNCITED_CHECK_IDS:
            assert check["citations"], f"{check['check_id']} has no citation"
            for key in check["citations"]:
                assert key in report["citations"]


def test_grade_json_output_file(sample_mcap, tmp_path):
    out = tmp_path / "report.json"
    result = CliRunner().invoke(main, ["grade", str(sample_mcap), "--json", "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    json.loads(out.read_text())  # valid JSON


def test_min_grade_gate_passes_when_good_enough(sample_mcap):
    result = CliRunner().invoke(main, ["grade", str(sample_mcap), "--json", "--min-grade", "C"])
    assert result.exit_code == 0


def _lerobot_with_dim_mismatch(tmp_path):
    """A minimal 2-episode LeRobot dataset where episode 1's state has a
    different dim than episode 0's -- triggers hygiene.schema_dim_consistency,
    a genuine DEFECT-class finding, so there's still a real dataset to exercise
    the --min-grade gate's *failing* path now that gripper-chatter/cross-modal-
    skew (the old fixture's only non-A findings) are RISK, not DEFECT."""
    import numpy as np
    import pandas as pd

    root = tmp_path / "dim_mismatch_lerobot"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)

    n = 20
    for ep_idx, dim in enumerate([4, 3]):
        state = np.zeros((n, dim))
        df = pd.DataFrame({
            "episode_index": ep_idx,
            "frame_index": np.arange(n),
            "timestamp": np.arange(n) * 0.02,
            "task_index": 0,
            "observation.state": list(state),
            "action": list(state.copy()),
        })
        df.to_parquet(root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")

    info = {
        "fps": 50,
        "total_episodes": 2,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [4], "names": ["j1", "j2", "j3", "gripper"]},
            "action": {"dtype": "float32", "shape": [4], "names": ["j1", "j2", "j3", "gripper"]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    return root


def test_min_grade_gate_fails_when_not_good_enough(tmp_path):
    root = _lerobot_with_dim_mismatch(tmp_path)
    result = CliRunner().invoke(main, ["grade", str(root), "--json", "--min-grade", "A"])
    assert result.exit_code == 1


def test_min_grade_gate_passes_at_a_when_only_risk_findings_fire(sample_mcap):
    """sample_mcap deliberately hits WARN/FAIL on gripper-chatter and
    cross-modal-skew (see tests/fixtures/scenario.py) -- both RISK-class since
    the grade_schema 1.0.0 retype, so an otherwise-clean dataset with only
    those findings must still clear --min-grade A."""
    result = CliRunner().invoke(main, ["grade", str(sample_mcap), "--json", "--min-grade", "A"])
    assert result.exit_code == 0


def test_unknown_format_reports_clean_error(tmp_path):
    bad = tmp_path / "mystery.xyz"
    bad.write_text("nope")
    result = CliRunner().invoke(main, ["grade", str(bad)])
    assert result.exit_code == 2
    assert "error:" in result.output


def test_share_report_dry_run_writes_payload_and_warns(sample_mcap, tmp_path):
    payload = tmp_path / "payload.json"
    result = CliRunner().invoke(
        main, ["grade", str(sample_mcap), "--json", "--share-report", "--share-report-dry-run", str(payload)]
    )
    assert result.exit_code == 0
    assert payload.exists()
    assert "not implemented" in result.output.lower()


def test_report_is_a_self_assessment_with_calibration_verdict(sample_lerobot):
    result = CliRunner().invoke(main, ["grade", str(sample_lerobot), "--json"])
    report = json.loads(result.output)
    assert report["report_type"] == "self-assessment"
    assert "not reviewed, signed, or attested" in report["self_assessment_note"]
    assert report["calibration"]["verdict"]
    assert report["calibration"]["detail"]
    # No honor-system trust marks anywhere in the free tier.
    assert "verified" not in report["overall"]
    assert "badge_markdown" not in report
    assert "Deepen Verified" not in json.dumps(report)


def test_error_goes_to_stderr_not_stdout(tmp_path):
    """A pilot run captured tqdm's stderr progress bars as "the error" because
    the actual error had gone to stdout instead -- errors must be on stderr,
    stdout must stay clean, regardless of --json."""
    bad = tmp_path / "mystery.xyz"
    bad.write_text("nope")
    result = CliRunner().invoke(main, ["grade", str(bad)])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "error:" in result.stderr


def test_json_error_emits_single_line_json_on_stderr(tmp_path):
    bad = tmp_path / "mystery.xyz"
    bad.write_text("nope")
    result = CliRunner().invoke(main, ["grade", str(bad), "--json"])
    assert result.exit_code == 2
    assert result.stdout == ""
    last_line = result.stderr.strip().splitlines()[-1]
    payload = json.loads(last_line)  # must be exactly one parseable JSON object
    assert payload["exit_code"] == 2
    assert payload["error"]


def test_quiet_disables_hf_progress_bars_env_var(sample_mcap, monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    result = CliRunner().invoke(main, ["grade", str(sample_mcap), "--quiet"])
    assert result.exit_code == 0
    import os

    assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"


def test_quiet_suppresses_output_only_note(sample_mcap, tmp_path):
    out = tmp_path / "ignored.json"
    quiet_result = CliRunner().invoke(main, ["grade", str(sample_mcap), "-o", str(out), "--quiet"])
    loud_result = CliRunner().invoke(main, ["grade", str(sample_mcap), "-o", str(out)])
    assert "note:" not in quiet_result.output
    assert "note:" in loud_result.output


def test_sample_flag_reports_sampling_block_in_json(sample_lerobot):
    result = CliRunner().invoke(main, ["grade", str(sample_lerobot), "--json", "--sample", "1", "--seed", "3"])
    report = json.loads(result.output)
    assert report["sampling"] == {"mode": "sample", "n": 1, "seed": 3, "episodes_total": 2}
    assert len(report["episodes"]) == 1


def test_max_episodes_flag_reports_sampling_block_in_json(sample_lerobot):
    result = CliRunner().invoke(main, ["grade", str(sample_lerobot), "--json", "--max-episodes", "1"])
    report = json.loads(result.output)
    assert report["sampling"] == {"mode": "head", "n": 1, "seed": None, "episodes_total": 2}


def test_no_sampling_flags_omit_sampling_key(sample_lerobot):
    result = CliRunner().invoke(main, ["grade", str(sample_lerobot), "--json"])
    report = json.loads(result.output)
    assert "sampling" not in report


def test_terminal_report_shows_sample_banner(sample_lerobot):
    result = CliRunner().invoke(main, ["grade", str(sample_lerobot), "--max-episodes", "1"])
    assert result.exit_code == 0
    assert "GRADED ON A SAMPLE" in result.output


def test_output_file_is_written_atomically_and_is_final(sample_lerobot, tmp_path):
    out = tmp_path / "report.json"
    result = CliRunner().invoke(main, ["grade", str(sample_lerobot), "--json", "-o", str(out)])
    assert result.exit_code == 0
    report = json.loads(out.read_text())
    assert report["partial"] is False
    # no leftover atomic-write tmp files
    assert not list(tmp_path.glob("*.tmp-*"))


# --- QA defect 3: --sample/--max-episodes of 0 or negative ------------------


@pytest.mark.parametrize("flag,value", [
    ("--sample", "0"), ("--sample", "-1"),
    ("--max-episodes", "0"), ("--max-episodes", "-3"),
])
def test_sample_and_max_episodes_reject_zero_and_negative(sample_lerobot, flag, value):
    result = CliRunner().invoke(main, ["grade", str(sample_lerobot), flag, value])
    assert result.exit_code == 2
    assert result.stdout == ""
    # rich highlights the standalone "1" with its own ANSI codes, so match
    # around it rather than the literal "must be >= 1" substring.
    assert flag in result.stderr and "must be >=" in result.stderr


def test_sample_zero_json_error_is_actionable(sample_lerobot):
    result = CliRunner().invoke(main, ["grade", str(sample_lerobot), "--json", "--sample", "0"])
    assert result.exit_code == 2
    last_line = result.stderr.strip().splitlines()[-1]
    payload = json.loads(last_line)
    assert payload["exit_code"] == 2
    assert "--sample must be >= 1" in payload["error"]


def test_positive_sample_and_max_episodes_still_work(sample_lerobot):
    # Regression guard: the new >= 1 validation must not reject legitimate values.
    assert CliRunner().invoke(main, ["grade", str(sample_lerobot), "--sample", "1"]).exit_code == 0
    assert CliRunner().invoke(main, ["grade", str(sample_lerobot), "--max-episodes", "1"]).exit_code == 0


# --- QA finding 1: NOT_ASSESSED floor, end-to-end via a real on-disk dataset -


def _garbage_lerobot_dataset(tmp_path):
    """A single, 2-frame LeRobot episode with dead-constant state -- the real
    on-disk ingest path for the NOT_ASSESSED floor: too few frames for
    topic_frequency_and_drops to analyze (MIN_MESSAGES_FOR_FREQ_ANALYSIS is 3),
    only 1 episode for schema_dim_consistency, and LeRobot has no tf concept at
    all -- every DEFECT-eligible check abstains. The constant state also trips
    plausibility.robot_data ("state values never change")."""
    import numpy as np
    import pandas as pd

    root = tmp_path / "garbage_lerobot"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)

    n = 2
    state = np.zeros((n, 3))
    df = pd.DataFrame({
        "episode_index": 0,
        "frame_index": np.arange(n),
        "timestamp": np.arange(n) * 0.02,
        "task_index": 0,
        "observation.state": list(state),
        "action": list(state.copy()),
    })
    df.to_parquet(root / "data" / "chunk-000" / "episode_000000.parquet")

    info = {
        "fps": 50,
        "total_episodes": 1,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [3], "names": ["j1", "j2", "j3"]},
            "action": {"dtype": "float32", "shape": [3], "names": ["j1", "j2", "j3"]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    return root


def test_garbage_dataset_end_to_end_via_cli_is_not_assessed(tmp_path):
    root = _garbage_lerobot_dataset(tmp_path)
    result = CliRunner().invoke(main, ["grade", str(root), "--json"])
    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["overall"] == {"score": 0, "grade": "NOT ASSESSED"}
    assert report["defect_classes"] == {}
    assert report["content_plausibility"]["flagged"] is True
    assert "never change" in report["content_plausibility"]["detail"]


def test_garbage_dataset_fails_min_grade_gate_at_any_bar(tmp_path):
    root = _garbage_lerobot_dataset(tmp_path)
    # Even the loosest possible bar (F) must not be "cleared" by a dataset
    # that couldn't be graded at all.
    result = CliRunner().invoke(main, ["grade", str(root), "--json", "--min-grade", "F"])
    assert result.exit_code == 1


def test_garbage_dataset_terminal_shows_plausibility_flag_and_claim_column(tmp_path):
    root = _garbage_lerobot_dataset(tmp_path)
    result = CliRunner().invoke(main, ["grade", str(root)])
    assert result.exit_code == 0
    assert "CONTENT PLAUSIBILITY FLAG" in result.output
    assert "NOT ASSESSED" in result.output
    assert "Claim" in result.output  # per-check claim-type column (QA finding 2)
