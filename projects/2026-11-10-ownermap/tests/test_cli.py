import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "samples"))

from regenerate_expected import CASES, run  # noqa: E402


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_outputs(name, monkeypatch):
    monkeypatch.chdir(ROOT)
    _code, out = run(CASES[name])
    with open(os.path.join(ROOT, "samples", "expected", name), encoding="utf-8") as fh:
        assert out == fh.read()


def test_exit_codes(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert run(CASES["lint.txt"])[0] == 1          # sample has errors
    assert run(CASES["review.txt"])[0] == 0
    assert run(["review", "-c", "samples/CODEOWNERS", "--registry", "samples/registry.json",
                "--author", "frank", "services/worker/jobs.py"])[0] == 1  # only the author owns it
    assert run(["coverage", "-c", "samples/CODEOWNERS", "--files", "samples/files.txt",
                "--min-coverage", "99"])[0] == 1


def test_root_walk(tmp_path, monkeypatch):
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "a.ts").write_text("x")
    (tmp_path / "CODEOWNERS").write_text("/web/ @dana\n")
    code, out = run(["coverage", "-c", str(tmp_path / "CODEOWNERS"), "--root", str(tmp_path)])
    assert "web/" in out and "TOTAL" in out and code == 0
