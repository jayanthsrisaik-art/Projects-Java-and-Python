import json
from pathlib import Path

from dqwatch.cli import main

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_series_sample_matches_expected_output(capsys):
    assert main(["series", "--csv", str(SAMPLES / "null_rate_series.csv"), "--metric", "null_rate:email"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got == json.loads((SAMPLES / "null_rate_expected_alerts.json").read_text())


def test_end_to_end_demo_matches_sample(tmp_path, capsys):
    db = str(tmp_path / "demo.db")
    assert main(["generate", "--db", db, "--days", "120", "--seed", "7", "--truth", str(tmp_path / "t.json"),
                 "--config-out", str(tmp_path / "c.json")]) == 0
    assert json.loads((tmp_path / "t.json").read_text()) == json.loads((SAMPLES / "demo_truth_seed7.json").read_text())
    assert json.loads((tmp_path / "c.json").read_text()) == json.loads((SAMPLES / "config.json").read_text())
    assert main(["profile", "--db", db, "--config", str(SAMPLES / "config.json")]) == 0
    capsys.readouterr()
    assert main(["check", "--db", db, "--config", str(SAMPLES / "config.json"), "--since", "2025-02-17"]) == 0
    assert capsys.readouterr().out == (SAMPLES / "demo_check_seed7.txt").read_text()
    # every injected incident in the demo is caught by at least one alert on one of its days
    assert main(["check", "--db", db, "--config", str(SAMPLES / "config.json"), "--since", "2025-02-17",
                 "--json"]) == 0
    alerts = {(a["table"], a["date"], a["metric"]) for a in json.loads(capsys.readouterr().out)}
    truth = json.loads((SAMPLES / "demo_truth_seed7.json").read_text())
    for inc in truth["incidents"]:
        assert any((inc["table"], d, m) in alerts for d in inc["dates"] for m in inc["metrics"]), inc
    # gate mode: exit 1 when the checked date has alerts, 0 when it is clean
    assert main(["check", "--db", db, "--config", str(SAMPLES / "config.json"), "--date", "2025-03-16",
                 "--fail-on-alert"]) == 1
    assert main(["check", "--db", db, "--config", str(SAMPLES / "config.json"), "--date", "2025-03-15",
                 "--fail-on-alert"]) == 0


def test_errors_exit_2(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"tables": [{"name": "t; drop", "partition_column": "pd"}]}))
    assert main(["profile", "--db", str(tmp_path / "x.db"), "--config", str(bad)]) == 2
    assert "invalid SQL identifier" in capsys.readouterr().err
    empty_cfg = tmp_path / "c.json"
    empty_cfg.write_text(json.dumps({"tables": [{"name": "t", "partition_column": "pd"}]}))
    assert main(["check", "--db", str(tmp_path / "x.db"), "--config", str(empty_cfg)]) == 2
