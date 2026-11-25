"""Regenerate samples/ and samples/expected_outputs.json (run from the project root)."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "samples"
BATCHES = {
    # a postcode-like field: only 0-3 occur
    "batch_form_b": ("3,3,2,2,0,0,0,0,0,0", 7),
    # a form that matches the training mix
    "batch_balanced": ("1,1,1,1,1,1,1,1,1,1", 8),
}


def run(*args):
    out = subprocess.run([sys.executable, "-m", "priorshift", *args], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout
    return out


def main():
    S.mkdir(exist_ok=True)
    expected = {"model": {"seed": 0, "factor": 2}}
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "model.joblib"
        run("train", "--out", str(model), "--seed", "0", "--factor", "2")
        for name, (priors, bseed) in BATCHES.items():
            run("make-batch", "--priors", priors, "--n", "300", "--batch-seed", str(bseed),
                "--out", str(S / f"{name}.csv"), "--labels", str(S / f"{name}_labels.csv"))
            expected[name] = json.loads(run("estimate", "--model", str(model), "--batch", str(S / f"{name}.csv"),
                                            "--labels", str(S / f"{name}_labels.csv")))
    (S / "expected_outputs.json").write_text(json.dumps(expected, indent=2) + "\n")
    print("samples regenerated")


if __name__ == "__main__":
    main()
