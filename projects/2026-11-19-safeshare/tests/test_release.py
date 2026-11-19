import csv
import datetime as dt
import json

import pytest

from safeshare.anonymize import Infeasible
from safeshare.policy import load_policy
from safeshare.pseudonym import Pseudonymizer
from safeshare.release import run_release
from conftest import POLICY

KEY = b"test-key-0123456789abcdef"


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def released(clinic, tmp_path_factory):
    out = tmp_path_factory.mktemp("rel") / "out"
    rep = run_release(clinic[0], load_policy(POLICY), out, KEY, "mondrian")
    return out, rep


def test_no_identifier_text_in_any_output_file(clinic, released):
    out, _ = released
    src = _rows(clinic[0] / "patients.csv")
    blob = "".join(p.read_text() for p in out.glob("*.csv"))
    for r in src[:200]:
        assert r["patient_id"] not in blob
        assert r["email"] not in blob
        assert r["phone"] not in blob
        assert r["name"] not in blob
    enc = _rows(clinic[0] / "encounters.csv")
    assert not any(e["encounter_id"] in blob for e in enc[:200])


def test_columns_and_verification(released):
    out, rep = released
    assert list(_rows(out / "patients.csv")[0]) == ["patient_id", "dob", "sex", "zip",
                                                     "plan", "diagnosis"]
    assert list(_rows(out / "encounters.csv")[0]) == ["encounter_id", "patient_id",
                                                       "visit_date", "cost"]
    assert rep["verified"]["k_achieved"] >= 5
    assert rep["tables"]["encounters"]["rows_in"] == rep["tables"]["encounters"]["rows_out"]
    assert json.loads((out / "release_report.json").read_text()) == rep


def test_referential_integrity_and_interval_preservation(clinic, released):
    out, _ = released
    pz = Pseudonymizer(KEY)
    pats = {r["patient_id"] for r in _rows(out / "patients.csv")}
    rel_enc = {r["encounter_id"]: r for r in _rows(out / "encounters.csv")}
    assert all(r["patient_id"] in pats for r in rel_enc.values())
    by_patient = {}
    for e in _rows(clinic[0] / "encounters.csv"):
        r = rel_enc[pz.token("encounter", e["encounter_id"])]
        assert r["patient_id"] == pz.token("patient", e["patient_id"])
        orig = dt.date.fromisoformat(e["visit_date"])
        new = dt.date.fromisoformat(r["visit_date"])
        by_patient.setdefault(e["patient_id"], set()).add((new - orig).days)
    # one constant, non-zero offset per patient -> intervals preserved exactly
    assert all(len(offs) == 1 and 0 not in offs for offs in by_patient.values())
    assert len({next(iter(o)) for o in by_patient.values()}) > 20


def test_rows_are_sorted_not_in_input_order(released):
    out, _ = released
    rows = _rows(out / "patients.csv")
    ids = [r["patient_id"] for r in rows]
    assert ids == sorted(ids)


def test_deterministic_and_key_dependent(clinic, released, tmp_path):
    out, _ = released
    pol = load_policy(POLICY)
    again = tmp_path / "again"
    run_release(clinic[0], pol, again, KEY, "mondrian")
    for name in ("patients.csv", "encounters.csv"):
        assert (again / name).read_bytes() == (out / name).read_bytes()
    other = tmp_path / "other"
    run_release(clinic[0], pol, other, b"another-key-0123456789", "mondrian")
    a = {r["patient_id"] for r in _rows(out / "patients.csv")}
    b = {r["patient_id"] for r in _rows(other / "patients.csv")}
    assert not a & b


def test_suppressed_subjects_vanish_from_child_tables(clinic, tmp_path):
    pol = load_policy(POLICY, k=25, l=4)
    pol.suppression_limit = 0.3
    out = tmp_path / "fd"
    rep = run_release(clinic[0], pol, out, KEY, "full_domain")
    pats = {r["patient_id"] for r in _rows(out / "patients.csv")}
    enc = _rows(out / "encounters.csv")
    assert all(e["patient_id"] in pats for e in enc)
    assert rep["subjects"] - rep["suppressed_subjects"] == len(pats)
    if rep["suppressed_subjects"]:
        assert rep["tables"]["encounters"]["rows_out"] < rep["tables"]["encounters"]["rows_in"]
    assert rep["verified"]["k_achieved"] >= 25 and rep["verified"]["l_achieved"] >= 4


def test_existing_output_kept_when_release_fails(clinic, released, tmp_path):
    out, _ = released
    before = (out / "patients.csv").read_bytes()
    with pytest.raises(Infeasible):
        run_release(clinic[0], load_policy(POLICY, k=10_000), out, KEY, "mondrian")
    assert (out / "patients.csv").read_bytes() == before


def test_replaces_previous_release_atomically(clinic, tmp_path):
    out = tmp_path / "out"
    pol = load_policy(POLICY)
    run_release(clinic[0], pol, out, KEY, "none")
    (out / "stale.csv").write_text("x\n")
    run_release(clinic[0], pol, out, KEY, "mondrian")
    assert not (out / "stale.csv").exists()
    assert json.loads((out / "release_report.json").read_text())["method"] == "mondrian"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]
