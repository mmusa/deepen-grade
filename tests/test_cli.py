# SPDX-License-Identifier: Apache-2.0
import json

from click.testing import CliRunner

from deepen_grade.cli import main


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
    for episode in report["episodes"]:
        for check in episode["checks"]:
            if check["severity"] != "n/a":
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
