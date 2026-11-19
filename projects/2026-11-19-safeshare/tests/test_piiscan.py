import pytest

from safeshare.piiscan import mask, scan
from safeshare.policy import load_policy
from safeshare.release import PIIFound, read_tables, run_release
from conftest import ROOT

LEAKY = ROOT / "samples" / "policy_leaky_notes.json"


def test_clean_policy_has_no_findings(clinic, policy):
    assert scan(read_tables(clinic[0], policy), policy) == []


def test_leaky_notes_are_caught_without_echoing_values(clinic):
    pol = load_policy(LEAKY)
    tables = read_tables(clinic[0], pol)
    findings = scan(tables, pol)
    rules = {(f["column"], f["rule"]) for f in findings}
    assert ("notes", "phone") in rules
    assert ("notes", "known_identifier") in rules
    header, rows = tables["patients"]
    phones = {r[header.index("phone")] for r in rows}
    names = {w.lower() for r in rows for w in r[header.index("name")].split()}
    for f in findings:
        for ex in f["examples"]:
            assert ex not in phones and ex.lower() not in names
            assert "*" in ex


def test_release_refuses_and_leaves_no_output(clinic, tmp_path):
    pol = load_policy(LEAKY)
    out = tmp_path / "out"
    with pytest.raises(PIIFound):
        run_release(clinic[0], pol, out, b"x" * 32)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []      # temp dir cleaned up


def test_allow_pii_acknowledges_a_column(clinic, policy_doc, tmp_path):
    import json
    policy_doc["tables"]["encounters"]["columns"]["notes"] = {"action": "keep",
                                                              "allow_pii": True}
    p = tmp_path / "pol.json"
    p.write_text(json.dumps(policy_doc))
    pol = load_policy(p)
    assert scan(read_tables(clinic[0], pol), pol) == []


def test_free_text_drop_column_is_not_an_identifier_source(clinic, policy):
    # notes (dropped) share words like "asthma" with the released diagnosis
    # column; that must not be flagged as an identifier leak
    tables = read_tables(clinic[0], policy)
    assert scan(tables, policy) == []


def test_mask():
    assert mask("617-555-0101") == "6**********1"
    assert mask("ab") == "**"
