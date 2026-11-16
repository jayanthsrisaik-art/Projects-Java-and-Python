"""Dynamic lineage oracle: run the models in SQLite, perturb one source column at a time and
record which model columns change.

A model column "changed" if the multiset of its values differs from the unperturbed run.
Every change observed this way is a real dependency on that database, so it gives a lower
bound on the true lineage. The static analysis is *sound* on the sample if every observed
dependency is in its (value ∪ control) set.

Several perturbations per column, each seeded:

* ``shuffle``  – permute the column's values across rows (multiset preserved; catches joins,
  grouping and row-wise expressions, not plain global aggregates)
* ``resample`` – replace half the rows with another value drawn from the column's own domain
  (moves rows across filters like ``status = 'complete'``)
* ``shift``    – numbers: add a non-zero offset; text: append a suffix (breaks equality joins,
  makes ``date()`` return NULL)
* ``nullify``  – set 30% of the rows to NULL
"""

from __future__ import annotations

import random
import sqlite3
from collections import Counter
from dataclasses import dataclass

from .project import ModelResult, Project, SourceTable

PERTURBATIONS = ("shuffle", "resample", "shift", "nullify")


# -- synthetic data ------------------------------------------------------------------------------
_FIRST = ["ana", "ben", "chen", "dara", "eli", "fatima", "goran", "hana", "ivan", "jaya", "kofi", "lena"]
_LAST = ["ng", "silva", "okafor", "berg", "rossi", "kim", "patel", "novak", "haddad", "mori"]


def _date(rng: random.Random, lo: str, hi: str) -> str:
    from datetime import date, timedelta

    a, b = date.fromisoformat(lo), date.fromisoformat(hi)
    return (a + timedelta(days=rng.randint(0, (b - a).days))).isoformat()


def generate_rows(project: Project, seed: int = 0) -> dict[str, list[dict]]:
    """Rows per source table from the generator hints in sources.json (pk, fk, values,
    range, date, pattern, null_rate). Tables are generated parents-first."""
    rng = random.Random(seed)
    data: dict[str, list[dict]] = {}
    pending = list(project.sources.values())
    while pending:
        progressed = False
        for t in list(pending):
            fks = [s["fk"].split(".")[0] for s in t.columns.values() if "fk" in s]
            if all(f in data or f == t.name for f in fks):
                data[t.name] = _gen_table(t, data, rng)
                pending.remove(t)
                progressed = True
        if not progressed:
            raise ValueError("foreign-key cycle in sources.json: " + ", ".join(t.name for t in pending))
    return data


def _gen_table(t: SourceTable, data, rng: random.Random) -> list[dict]:
    rows = []
    for i in range(t.rows):
        row = {}
        for c, s in t.columns.items():
            if s.get("pk"):
                v = i + 1
            elif "fk" in s:
                pt, pc = s["fk"].split(".")
                v = rng.choice(data[pt])[pc]
            elif "values" in s:
                v = rng.choice(s["values"])
            elif "date" in s:
                v = _date(rng, *s["date"])
            elif "range" in s:
                lo, hi = s["range"]
                v = rng.randint(lo, hi) if s["type"] == "INTEGER" else round(rng.uniform(lo, hi), 2)
            elif s.get("pattern") == "email":
                v = f"{rng.choice(_FIRST)}.{rng.choice(_LAST)}{i}@example.com"
            elif s.get("pattern") == "name":
                v = f"{rng.choice(_FIRST)} {rng.choice(_LAST)} {i}"
            elif s["type"] == "TEXT":
                v = f"{c}_{rng.randint(1, 20)}"
            else:
                v = rng.randint(0, 100)
            if s.get("null_rate") and rng.random() < s["null_rate"]:
                v = None
            row[c] = v
        rows.append(row)
    return rows


# -- database ------------------------------------------------------------------------------------
def build_db(project: Project, data: dict[str, list[dict]], results: dict[str, ModelResult]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    for t in project.sources.values():
        cols = ", ".join(f'"{c}" {s["type"]}' for c, s in t.columns.items())
        conn.execute(f'CREATE TABLE "{t.name}" ({cols})')
        names = list(t.columns)
        conn.executemany(
            f'INSERT INTO "{t.name}" VALUES ({", ".join("?" for _ in names)})',
            [tuple(r[c] for c in names) for r in data[t.name]],
        )
    for name, r in results.items():  # results are in topological order
        if r.status == "ok":
            conn.execute(f'CREATE VIEW "{name}" AS {r.sql.strip().rstrip(";")}')
    conn.commit()
    return conn


def snapshot(conn: sqlite3.Connection, results: dict[str, ModelResult]) -> dict[str, Counter]:
    """model.col -> Counter of values. Also checks that SQLite's output column names match
    the statically inferred schema."""
    snap: dict[str, Counter] = {}
    for name, r in results.items():
        if r.status != "ok":
            continue
        cur = conn.execute(f'SELECT * FROM "{name}"')
        got = [d[0].lower() for d in cur.description]
        if got != list(r.columns):
            raise AssertionError(f"{name}: SQLite columns {got} != inferred {list(r.columns)}")
        rows = cur.fetchall()
        for i, c in enumerate(got):
            snap[f"{name}.{c}"] = Counter(_norm(row[i]) for row in rows)
    return snap


def _norm(v):
    return round(v, 9) if isinstance(v, float) else v


def _perturb(
    conn: sqlite3.Connection, table: str, col: str, spec: dict, how: str, rng: random.Random
) -> bool:
    """Apply one perturbation in place. Primary keys stay unique and non-null (the truth
    labels assume declared keys hold), so they get a permutation of a random subset or a
    constant shift instead of resample/nullify. Returns False if the perturbation was skipped."""
    ctype = spec["type"]
    rows = conn.execute(f'SELECT rowid, "{col}" FROM "{table}"').fetchall()
    ids = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    if spec.get("pk"):
        if how == "nullify":
            return False
        if how == "resample":  # permute a random half of the keys among themselves
            idx = [i for i in range(len(vals)) if rng.random() < 0.5]
            sub = [vals[i] for i in idx]
            rng.shuffle(sub)
            new = vals[:]
            for i, v in zip(idx, sub):
                new[i] = v
        elif how == "shift":
            new = [v + 10_000 for v in vals]
        else:
            new = vals[:]
            rng.shuffle(new)
        conn.executemany(f'UPDATE "{table}" SET "{col}" = ? WHERE rowid = ?', list(zip(new, ids)))
        return True
    if how == "shuffle":
        new = vals[:]
        rng.shuffle(new)
    elif how == "resample":
        domain = sorted({v for v in vals if v is not None}, key=repr)
        new = [
            rng.choice([d for d in domain if d != v] or domain) if domain and rng.random() < 0.5 else v
            for v in vals
        ]
    elif how == "shift":
        if ctype == "TEXT":
            new = [None if v is None else f"{v}~" for v in vals]
        else:
            new = [None if v is None else v + rng.choice([1, 2, 3, 7]) for v in vals]
    elif how == "nullify":
        new = [None if rng.random() < 0.3 else v for v in vals]
    else:
        raise ValueError(how)
    conn.executemany(f'UPDATE "{table}" SET "{col}" = ? WHERE rowid = ?', list(zip(new, ids)))
    return True


@dataclass
class OracleResult:
    observed: dict[str, set[str]]  # model column -> source columns whose perturbation changed it
    by_perturbation: dict[str, dict[str, set[str]]]  # source col -> perturbation -> changed cols


def run_oracle(project: Project, results: dict[str, ModelResult], seeds=(0, 1, 2)) -> OracleResult:
    observed: dict[str, set[str]] = {
        f"{m}.{c}": set() for m, r in results.items() if r.status == "ok" for c in r.columns
    }
    by_pert: dict[str, dict[str, set[str]]] = {}
    for seed in seeds:
        data = generate_rows(project, seed)
        base_conn = build_db(project, data, results)
        base = snapshot(base_conn, results)
        for t in project.sources.values():
            for c, spec in t.columns.items():
                src = f"{t.name}.{c}"
                for k, how in enumerate(PERTURBATIONS):
                    conn = sqlite3.connect(":memory:")
                    base_conn.backup(conn)
                    if not _perturb(conn, t.name, c, spec, how, random.Random(seed * 1000 + k)):
                        conn.close()
                        continue
                    snap = snapshot(conn, results)
                    conn.close()
                    changed = {node for node, counts in snap.items() if counts != base[node]}
                    by_pert.setdefault(src, {}).setdefault(how, set()).update(changed)
                    for node in changed:
                        observed[node].add(src)
        base_conn.close()
    return OracleResult(observed, by_pert)
