import copy
import shutil

import pytest

from safeshare.policy import PolicyError, parse_policy
from safeshare.release import read_tables


def _bad(doc, mutate):
    d = copy.deepcopy(doc)
    mutate(d)
    with pytest.raises(PolicyError):
        parse_policy(d)


def test_sample_policy_parses(policy):
    assert policy.k == 5 and policy.subject_domain == "patient"
    assert [r.column for r in policy.qi_rules()] == ["dob", "sex", "zip"]


def test_invalid_policies(policy_doc):
    t = "tables"
    _bad(policy_doc, lambda d: d[t]["patients"]["columns"]["name"].update(action="hide"))
    _bad(policy_doc, lambda d: d[t]["encounters"]["columns"].update(
        cost={"action": "qi", "type": "numeric"}))           # QI outside subject table
    _bad(policy_doc, lambda d: d[t]["encounters"]["columns"]["visit_date"].update(
        by="cost"))                                          # shift not by subject
    _bad(policy_doc, lambda d: d[t]["encounters"]["columns"]["visit_date"].update(
        domain="encounter"))
    _bad(policy_doc, lambda d: d[t]["patients"]["columns"].update(
        diagnosis={"action": "keep"}, ) or d.update(l=2))   # l>1 needs sensitive
    _bad(policy_doc, lambda d: d.update(k=1))
    _bad(policy_doc, lambda d: d.update(subject_key="name"))
    _bad(policy_doc, lambda d: d[t]["patients"]["columns"]["dob"].update(type="time"))
    _bad(policy_doc, lambda d: d[t]["patients"]["columns"]["name"].update(colour="red"))
    _bad(policy_doc, lambda d: d.update(suppression_limit=1.5))


def _copy(clinic, tmp_path):
    src, _ = clinic
    dst = tmp_path / "in"
    shutil.copytree(src, dst)
    return dst


def test_fail_closed_on_new_upstream_column(clinic, policy, tmp_path):
    d = _copy(clinic, tmp_path)
    p = d / "patients.csv"
    lines = p.read_text().splitlines()
    lines = [lines[0] + ",ssn"] + [ln + ",123-45-6789" for ln in lines[1:]]
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(PolicyError, match="undeclared column"):
        read_tables(d, policy)


def test_fail_closed_on_missing_column_and_extra_table(clinic, policy, tmp_path):
    d = _copy(clinic, tmp_path)
    (d / "audit_log.csv").write_text("who,what\nx,y\n")
    with pytest.raises(PolicyError, match="undeclared table"):
        read_tables(d, policy)
    (d / "audit_log.csv").unlink()
    p = d / "encounters.csv"
    rows = [ln.rsplit(",", 1)[0] for ln in p.read_text().splitlines()]
    p.write_text("\n".join(rows) + "\n")
    with pytest.raises(PolicyError):
        read_tables(d, policy)
