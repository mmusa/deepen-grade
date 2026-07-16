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


def test_min_grade_gate_fails_when_not_good_enough(sample_mcap):
    result = CliRunner().invoke(main, ["grade", str(sample_mcap), "--json", "--min-grade", "A"])
    assert result.exit_code == 1


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
