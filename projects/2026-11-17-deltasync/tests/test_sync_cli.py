import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from deltasync import sync_file
from deltasync.cli import main
from deltasync.signature import strong_hash

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
EXPECTED = SAMPLES / "expected"


def test_sync_updates_file_atomically_and_reports(tmp_path):
    dst = tmp_path / "copy.csv"
    dst.write_bytes((SAMPLES / "basis.csv").read_bytes())
    report = sync_file(SAMPLES / "target.csv", dst, block_size=256)
    assert dst.read_bytes() == (SAMPLES / "target.csv").read_bytes()
    assert report.method == "delta" and report.attempts == 1
    assert report.wire_bytes < 0.25 * report.target_len
    assert [p.name for p in tmp_path.iterdir()] == ["copy.csv"]   # no temp files left


def test_sync_missing_destination_and_noop(tmp_path):
    dst = tmp_path / "sub" / "new.csv"
    assert sync_file(SAMPLES / "basis.csv", dst).method == "whole-file"
    assert dst.read_bytes() == (SAMPLES / "basis.csv").read_bytes()
    mtime = os.stat(dst).st_mtime_ns
    assert sync_file(SAMPLES / "basis.csv", dst).method == "unchanged-noop"
    assert os.stat(dst).st_mtime_ns == mtime


def test_sync_dry_run_does_not_write(tmp_path):
    dst = tmp_path / "d.csv"
    dst.write_bytes((SAMPLES / "basis.csv").read_bytes())
    sync_file(SAMPLES / "target.csv", dst, block_size=256, dry_run=True)
    assert dst.read_bytes() == (SAMPLES / "basis.csv").read_bytes()


def test_sync_retries_with_full_strength_hash_after_collision(tmp_path):
    import random
    from deltasync.rollsum import weak_checksum
    rng = random.Random(9)
    seen = {}
    while True:
        blk = bytes(rng.getrandbits(8) for _ in range(4))
        key = (weak_checksum(blk), strong_hash(blk, 1))
        if key in seen and seen[key] != blk:
            x, y = seen[key], blk
            break
        seen[key] = blk
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.write_bytes(y)
    dst.write_bytes(x)
    report = sync_file(src, dst, block_size=4, strong_len=1)
    assert dst.read_bytes() == y
    assert report.method == "delta-retry" and report.attempts == 2


def test_cli_matches_expected_sample_outputs(tmp_path, capsys):
    sig = tmp_path / "b.sig"
    delta = tmp_path / "d.delta"
    out = tmp_path / "out.csv"
    assert main(["signature", str(SAMPLES / "basis.csv"), str(sig), "--block-size", "256"]) == 0
    assert main(["delta", str(sig), str(SAMPLES / "target.csv"), str(delta)]) == 0
    assert delta.read_bytes() == (EXPECTED / "basis_to_target.delta").read_bytes()
    assert main(["patch", str(SAMPLES / "basis.csv"), str(delta), str(out)]) == 0
    assert out.read_bytes() == (SAMPLES / "target.csv").read_bytes()
    capsys.readouterr()
    assert main(["inspect", str(delta)]) == 0
    assert capsys.readouterr().out == (EXPECTED / "inspect.txt").read_text()
    assert main(["chunks", str(SAMPLES / "log_v1.bin"), str(SAMPLES / "log_v2.bin"), "--avg", "4096"]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads((EXPECTED / "chunks.json").read_text())
    assert main(["sync", str(SAMPLES / "target.csv"), str(SAMPLES / "basis.csv"),
                 "--block-size", "256", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads((EXPECTED / "sync_dry_run.json").read_text())


def test_cli_patch_wrong_basis_fails_cleanly(tmp_path):
    delta = EXPECTED / "basis_to_target.delta"
    out = tmp_path / "out.csv"
    proc = subprocess.run([sys.executable, "-m", "deltasync", "patch", str(SAMPLES / "target.csv"),
                           str(delta), str(out)], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 2
    assert "hash mismatch" in proc.stderr
    assert list(tmp_path.iterdir()) == []   # neither output nor partial file left behind


def test_cli_rejects_garbage_delta(tmp_path):
    bad = tmp_path / "bad.delta"
    bad.write_bytes(b"not a delta")
    assert main(["inspect", str(bad)]) == 2
