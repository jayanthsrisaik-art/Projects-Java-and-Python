import json

from sqlcritic.cli import main

from .conftest import ROOT

SAMPLES = ROOT / "samples"


def _strip(d):
    d.pop("seconds", None)
    return d


def test_score_matches_expected(db_path, capsys):
    assert main(["--db", str(db_path), "score", str(SAMPLES / "predictions_example.jsonl")]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads((SAMPLES / "expected_score.json").read_text())


def test_ask_matches_expected(db_path, capsys):
    for name, q in [("expected_ask_dev_wording.json", "Show revenue by region for 2024."),
                    ("expected_ask_heldout_wording.json", "Break down 2024 sales across regions.")]:
        assert main(["--db", str(db_path), "ask", q]) == 0
        got = _strip(json.loads(capsys.readouterr().out))
        assert got == _strip(json.loads((SAMPLES / name).read_text())), name


def test_eval_gate_fails_on_regression(db_path, tmp_path, capsys):
    baseline = json.loads((ROOT / "results" / "eval_heuristic.json").read_text())
    baseline["modes"]["critic"]["dev"]["accuracy"] = 1.01       # impossible bar
    path = tmp_path / "b.json"
    path.write_text(json.dumps(baseline))
    rc = main(["--db", str(db_path), "eval", "--split", "dev", "--modes", "critic",
               "--baseline", str(path)])
    assert rc == 1
    assert "REGRESSION: critic/dev" in capsys.readouterr().err
