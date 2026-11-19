import json
import shutil

from safeshare.cli import main
from conftest import ROOT

SAMPLES = ROOT / "samples"


def test_exit_codes(clinic, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SAFESHARE_KEY", raising=False)
    data = str(clinic[0])
    pol = str(SAMPLES / "policy.json")
    leaky = str(SAMPLES / "policy_leaky_notes.json")
    assert main(["scan", "--input", data, "--policy", pol]) == 0
    assert main(["scan", "--input", data, "--policy", leaky]) == 3
    assert main(["release", "--input", data, "--policy", leaky, "--out",
                 str(tmp_path / "a"), "--demo-key"]) == 3
    assert main(["release", "--input", data, "--policy", pol, "--out",
                 str(tmp_path / "b")]) == 2            # no key supplied
    assert main(["release", "--input", data, "--policy", pol, "--out",
                 str(tmp_path / "c"), "--demo-key", "--k", "100000"]) == 4
    assert main(["release", "--input", data, "--policy", pol, "--out",
                 str(tmp_path / "d"), "--demo-key"]) == 0
    err = capsys.readouterr().err
    assert "refusing to release" in err and "no key" in err


def test_sample_release_matches_expected_output(tmp_path, capsys):
    out = tmp_path / "release"
    assert main(["release", "--input", str(SAMPLES / "clinic"), "--policy",
                 str(SAMPLES / "policy.json"), "--out", str(out), "--demo-key"]) == 0
    exp = SAMPLES / "expected" / "release"
    for name in ("patients.csv", "encounters.csv", "release_report.json"):
        assert (out / name).read_text() == (exp / name).read_text(), name


def test_sample_evaluation_matches_expected(tmp_path, capsys):
    rel = tmp_path / "release"
    shutil.copytree(SAMPLES / "expected" / "release", rel)
    capsys.readouterr()
    assert main(["evaluate", "--original", str(SAMPLES / "clinic"), "--release",
                 str(rel), "--voters", str(SAMPLES / "clinic_public" / "voters.csv"),
                 "--policy", str(SAMPLES / "policy.json"), "--demo-key"]) == 0
    got = json.loads(capsys.readouterr().out)
    exp = json.loads((SAMPLES / "expected" / "evaluate.json").read_text())
    assert got == exp
