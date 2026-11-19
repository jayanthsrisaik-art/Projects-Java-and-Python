"""Release pipeline: CSV tables in -> policy-checked, anonymised CSV tables out.

Steps, all-or-nothing:
1. read every ``<table>.csv`` in the input directory;
2. fail closed on undeclared / missing columns;
3. PII-scan the columns that would be copied verbatim (refuse on hits);
4. anonymise the subject table's quasi-identifiers (k / l);
5. drop suppressed subjects *and their rows in every child table*;
6. tokenise ids, shift dates per subject, drop identifiers;
7. sort rows by output values (input order can leak, e.g. signup order);
8. verify the released subject table really is k-anonymous / l-diverse by
   re-parsing the written cells, then atomically swap the output directory in.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

from . import anonymize, piiscan
from .policy import Policy, PolicyError
from .pseudonym import Pseudonymizer
from .qi import QISpec

METHODS = ("mondrian", "mondrian_tight", "full_domain", "none")


class PIIFound(RuntimeError):
    def __init__(self, findings):
        super().__init__(f"{len(findings)} PII finding(s) in verbatim columns")
        self.findings = findings


def read_tables(input_dir: str | Path, policy: Policy) -> dict:
    input_dir = Path(input_dir)
    tables = {}
    for table in policy.tables:
        path = input_dir / f"{table}.csv"
        if not path.exists():
            raise PolicyError(f"declared table {table!r} has no file {path}")
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise PolicyError(f"{path} is empty") from exc
            rows = [r for r in reader if r]
        for n, r in enumerate(rows, start=2):
            if len(r) != len(header):
                raise PolicyError(f"{path}:{n}: expected {len(header)} fields, "
                                  f"got {len(r)}")
        policy.check_columns(table, header)
        tables[table] = (header, rows)
    extra = sorted(p.stem for p in input_dir.glob("*.csv") if p.stem not in policy.tables)
    if extra:
        raise PolicyError(f"undeclared table file(s) in input: {extra} (fail-closed)")
    return tables


def subject_records(tables: dict, policy: Policy):
    """QI specs + records ``{"qi", "sens", "key"}`` for the subject table."""
    header, rows = tables[policy.subject_table]
    qrules = policy.qi_rules()
    srule = policy.sensitive_rule()
    qcols = [header.index(r.column) for r in qrules]
    scol = header.index(srule.column) if srule else None
    kcol = header.index(policy.subject_key)
    specs = [QISpec(r.column, r.kind) for r in qrules]
    encoded = [[s.encode(row[c]) for s, c in zip(specs, qcols)] for row in rows]
    for d, s in enumerate(specs):
        s.fit(e[d] for e in encoded)
    keys = [row[kcol] for row in rows]
    if len(set(keys)) != len(keys):
        raise PolicyError(f"{policy.subject_table}.{policy.subject_key} is not unique")
    records = [{"qi": tuple(e), "sens": row[scol] if scol is not None else None,
                "key": row[kcol]} for e, row in zip(encoded, rows)]
    return specs, records


def anonymise(records, specs, policy: Policy, method: str):
    if method in ("mondrian", "mondrian_tight"):
        return anonymize.mondrian(records, specs, policy.k, policy.l,
                                  tight=method == "mondrian_tight")
    if method == "full_domain":
        return anonymize.full_domain(records, specs, policy.k, policy.l,
                                     policy.suppression_limit)
    if method == "none":
        return anonymize.identity(records, specs)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def _transform_table(table, header, rows, policy, pz, subject_cells=None,
                     specs=None, suppressed_keys=frozenset()):
    rules = policy.rules(table)
    out_cols = [c for c in header if rules[c].action != "drop"]
    sdomain = policy.subject_domain
    subject_cols = [i for i, c in enumerate(header)
                    if rules[c].action == "token" and rules[c].domain == sdomain]
    qi_pos = {s.name: d for d, s in enumerate(specs or [])}
    out_rows = []
    for ri, row in enumerate(rows):
        if any(row[i] in suppressed_keys for i in subject_cols):
            continue
        rec = dict(zip(header, row))
        new = []
        for c in out_cols:
            rule = rules[c]
            v = rec[c]
            if rule.action == "token":
                new.append(pz.token(rule.domain, v) if v != "" else "")
            elif rule.action == "date_shift":
                new.append(pz.shift_date(rule.domain, rec[rule.by], v))
            elif rule.action == "qi":
                d = qi_pos[c]
                new.append(specs[d].format(subject_cells[ri][d]))
            else:                      # keep / sensitive
                new.append(v)
        out_rows.append(new)
    out_rows.sort()
    return out_cols, out_rows


def verify_release(out_dir: Path, policy: Policy, specs) -> dict:
    """Re-read the written subject table and recheck k / l from the text."""
    with (out_dir / f"{policy.subject_table}.csv").open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    qcols = [header.index(s.name) for s in specs]
    srule = policy.sensitive_rule()
    scol = header.index(srule.column) if srule else None
    groups: dict = {}
    for r in rows:
        box = tuple(s.parse(r[c]) for s, c in zip(specs, qcols))
        groups.setdefault(box, []).append(r[scol] if scol is not None else None)
    sizes = [len(g) for g in groups.values()]
    div = [len(set(g)) for g in groups.values()]
    return {"rows": len(rows), "classes": len(groups),
            "k_achieved": min(sizes) if sizes else 0,
            "l_achieved": min(div) if div and scol is not None else None}


def run_release(input_dir, policy: Policy, out_dir, key: bytes,
                method: str = "mondrian", demo_key: bool = False) -> dict:
    tables = read_tables(input_dir, policy)
    findings = piiscan.scan(tables, policy)
    if findings:
        raise PIIFound(findings)
    specs, records = subject_records(tables, policy)
    result = anonymise(records, specs, policy, method)
    suppressed = frozenset(records[i]["key"] for i in result.suppressed)
    pz = Pseudonymizer(key, policy.max_date_shift_days)

    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".safeshare-", dir=out_dir.parent))
    try:
        table_counts = {}
        for table, (header, rows) in tables.items():
            is_subject = table == policy.subject_table
            cols, out_rows = _transform_table(
                table, header, rows, policy, pz,
                subject_cells=result.cells if is_subject else None,
                specs=specs if is_subject else None,
                suppressed_keys=suppressed)
            with (tmp / f"{table}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh, lineterminator="\n")
                w.writerow(cols)
                w.writerows(out_rows)
            table_counts[table] = {"rows_in": len(rows), "rows_out": len(out_rows),
                                   "columns_out": cols}
        check = verify_release(tmp, policy, specs)
        if method != "none":
            if check["k_achieved"] < policy.k or (
                    policy.l > 1 and (check["l_achieved"] or 0) < policy.l):
                raise anonymize.Infeasible(f"post-release check failed: {check}")
        ncp = _avg_ncp(result, specs)
        report = {
            "method": method,
            "policy_digest": policy.digest,
            "k_target": policy.k, "l_target": policy.l,
            "demo_key": demo_key,
            "subjects": len(records),
            "suppressed_subjects": len(suppressed),
            "verified": check,
            "class_stats": anonymize.class_stats(result, records),
            "avg_ncp": round(ncp, 4),
            "full_domain_levels": (dict(zip((s.name for s in specs), result.level))
                                   if result.level else None),
            "tables": table_counts,
            "pii_findings": [],
        }
        (tmp / "release_report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                 encoding="utf-8")
        if out_dir.exists():
            old = out_dir.with_name(out_dir.name + ".old")
            shutil.rmtree(old, ignore_errors=True)
            os.replace(out_dir, old)
            os.replace(tmp, out_dir)
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.replace(tmp, out_dir)
        return report
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _avg_ncp(result, specs) -> float:
    n = len(result.cells)
    total = 0.0
    for cells in result.cells:
        if cells is None:
            total += len(specs)
        else:
            total += sum(s.ncp(c) for s, c in zip(specs, cells))
    return total / (n * len(specs)) if n else 0.0
