import json
from pathlib import Path

from asofqa.cli import main

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def test_generate_and_ask_json(tmp_path, capsys):
    assert main(["generate", str(tmp_path / "c"), "--seed", "5"]) == 0
    assert len(list((tmp_path / "c" / "docs").glob("*.md"))) > 20
    capsys.readouterr()
    main(["ask", str(tmp_path / "c" / "docs"), "What is the annual learning budget?", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["policy"] == "learning" and out["values"] and not out["abstain"]


def test_sample_answers_match_expected(capsys, monkeypatch):
    import io
    for strategy in ("asof", "latest", "naive"):
        monkeypatch.setattr("sys.stdin", io.StringIO((SAMPLES / "questions.txt").read_text()))
        main(["ask", str(SAMPLES / "corpus" / "docs"), "--strategy", strategy])
        assert capsys.readouterr().out == (SAMPLES / f"expected_answers_{strategy}.txt").read_text()


def test_sample_corpus_is_reproducible(tmp_path):
    main(["generate", str(tmp_path / "c"), "--seed", "7", "--split", "dev"])
    for p in (SAMPLES / "corpus" / "docs").glob("*.md"):
        assert (tmp_path / "c" / "docs" / p.name).read_text() == p.read_text()
    assert (tmp_path / "c" / "questions.jsonl").read_text() == (SAMPLES / "corpus" / "questions.jsonl").read_text()


def test_eval_writes_report(tmp_path, capsys):
    assert main(["eval", "--split", "dev", "--out", str(tmp_path)]) == 0
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert set(m["dev"]["strategies"]) == {"naive", "latest", "asof_nostatus", "asof"}
    assert (tmp_path / "report.md").read_text().startswith("# asofqa evaluation")


def test_options_may_precede_questions(capsys):
    main(["ask", str(SAMPLES / "corpus" / "docs"), "--ref-date", "2023-01-01", "What is the remote work allowance?"])
    assert "remote-v2" in capsys.readouterr().out


def test_unknown_option_is_rejected():
    import pytest
    with pytest.raises(SystemExit):
        main(["ask", str(SAMPLES / "corpus" / "docs"), "--bogus", "q"])
