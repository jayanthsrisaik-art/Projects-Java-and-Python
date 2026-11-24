import json
import os

import pytest

from rgatext.cli import main
from rgatext.sim import ScenarioError, run_scenario

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(os.path.dirname(HERE), "samples")


@pytest.mark.parametrize("name", ["concurrent_typing", "flaky_network"])
def test_simulate_samples_match_expected_output(name, capsys):
    code = main(["simulate", os.path.join(SAMPLES, f"{name}.json")])
    out = json.loads(capsys.readouterr().out)
    with open(os.path.join(SAMPLES, "expected", f"{name}.json")) as fh:
        expected = json.load(fh)
    assert code == 0
    assert out == expected
    assert out["crdt"]["converged"] and not out["naive"]["converged"]


def test_merge_sample_logs_matches_expected_text(tmp_path, capsys):
    out_log = tmp_path / "merged.jsonl"
    code = main(["merge", os.path.join(SAMPLES, "logs", "alice.jsonl"),
                 os.path.join(SAMPLES, "logs", "bob.jsonl"), "--out", str(out_log)])
    with open(os.path.join(SAMPLES, "expected", "merged.txt")) as fh:
        expected = fh.read()
    assert code == 0
    assert capsys.readouterr().out == expected
    # merging is order independent and the written log round-trips
    main(["merge", os.path.join(SAMPLES, "logs", "bob.jsonl"),
          os.path.join(SAMPLES, "logs", "alice.jsonl")])
    assert capsys.readouterr().out == expected
    main(["merge", str(out_log)])
    assert capsys.readouterr().out == expected


def test_merge_reports_incomplete_logs_with_exit_code_2(tmp_path, capsys):
    lines = open(os.path.join(SAMPLES, "logs", "bob.jsonl")).read().splitlines()
    partial = tmp_path / "partial.jsonl"
    partial.write_text("\n".join(lines[5:]) + "\n")  # drop the first ops
    assert main(["merge", str(partial)]) == 2
    assert "missing dependencies" in capsys.readouterr().err


def test_merge_rejects_invalid_json(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"kind": "ins"\n')
    assert main(["merge", str(bad)]) == 2
    assert "bad.jsonl:1" in capsys.readouterr().err


def test_fuzz_and_bench_commands(capsys):
    assert main(["fuzz", "--runs", "5", "--steps", "120"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["crdt_converged"] == summary["crdt_matches_oracle"] == 5
    assert main(["bench", "--chars", "500"]) == 0
    assert json.loads(capsys.readouterr().out)["converged"] is True


def test_bad_scenarios_raise_clear_errors(tmp_path, capsys):
    with pytest.raises(ScenarioError):
        run_scenario({"replicas": ["a", "a"], "steps": []})
    with pytest.raises(ScenarioError):
        run_scenario({"replicas": ["a"], "steps": [{"sync": ["a", "ghost"]}]})
    with pytest.raises(ScenarioError):
        run_scenario({"replicas": ["a"], "steps": [{"teleport": 1}]})
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"replicas": ["a"], "steps": [{"edit": "a"}]}))
    assert main(["simulate", str(path)]) == 2
    assert "edit needs" in capsys.readouterr().err
