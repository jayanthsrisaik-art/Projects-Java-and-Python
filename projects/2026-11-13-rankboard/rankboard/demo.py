"""Scripted two-user session against the real HTTP API (in-process TestClient).

    python -m rankboard.demo                 # prints the transcript
    python -m rankboard.demo --check         # compares with samples/scenario_expected.txt

Board and column ids are random, so the transcript refers to columns by name
and uses client-chosen card ids.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

with warnings.catch_warnings():  # newer Starlette nudges towards httpx2; httpx works fine
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from .api import create_app
from .store import replay_feed

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def run(scenario: dict) -> list[str]:
    app = create_app(":memory:", max_key_len=scenario["max_key_len"])
    http = TestClient(app)
    out: list[str] = []
    snap = http.post("/api/boards", json={"name": scenario["board"],
                                          "columns": scenario["columns"]}).json()
    board = snap["board"]["id"]
    col = {c["name"]: c["id"] for c in snap["columns"]}
    col_name = {v: k for k, v in col.items()}
    cursors = {"alice": snap["seq"], "bob": snap["seq"]}
    replicas = {"alice": {}, "bob": {}}
    versions: dict[str, int] = {}

    def column_order(name: str) -> str:
        s = http.get(f"/api/boards/{board}").json()
        ids = [c["id"] for c in s["cards"] if c["column_id"] == col[name]]
        return " ".join(ids) or "(empty)"

    for step in scenario["steps"]:
        who, op = step["who"], step["op"]
        if op == "create":
            r = http.post(f"/api/boards/{board}/cards", json={
                "id": step["id"], "column_id": col[step["column"]], "title": step["title"],
                "after_id": step.get("after"), "before_id": step.get("before")})
            body = r.json()
            versions[step["id"]] = body["card"]["version"]
            out.append(f"{who:5} create {step['id']:<8} -> {r.status_code} rank={body['card']['rank']}"
                       f" placement={body['placement']} rebalanced={body['rebalanced']}")
        elif op == "move":
            r = http.post(f"/api/cards/{step['id']}/move", json={
                "column_id": col[step["column"]], "after_id": step.get("after"),
                "before_id": step.get("before")})
            body = r.json()
            versions[step["id"]] = body["card"]["version"]
            out.append(f"{who:5} move   {step['id']:<8} -> {r.status_code} {step['column']}"
                       f" rank={body['card']['rank']} placement={body['placement']}")
        elif op == "edit":
            v = step.get("version", versions.get(step["id"]))
            r = http.patch(f"/api/cards/{step['id']}", headers={"If-Match": f'"{v}"'},
                           json={"title": step["title"]})
            body = r.json()
            if r.status_code == 200:
                versions[step["id"]] = body["version"]
                out.append(f"{who:5} edit   {step['id']:<8} v{v} -> 200 now v{body['version']}")
            else:
                out.append(f"{who:5} edit   {step['id']:<8} v{v} -> {r.status_code} "
                           f"current v{body['current']['version']} title={body['current']['title']!r}")
        elif op == "delete":
            v = versions[step["id"]]
            r = http.delete(f"/api/cards/{step['id']}", headers={"If-Match": f'"{v}"'})
            out.append(f"{who:5} delete {step['id']:<8} -> {r.status_code}")
        elif op == "sync":
            r = http.get(f"/api/boards/{board}/changes", params={"since": cursors[who]})
            if r.status_code == 410:
                out.append(f"{who:5} sync since {cursors[who]} -> 410, reloading snapshot")
                s = http.get(f"/api/boards/{board}").json()
                replicas[who] = {c["id"]: c for c in s["cards"]}
                cursors[who] = s["seq"]
                continue
            body = r.json()
            replicas[who] = replay_feed(body["changes"], replicas[who])
            reasons: dict[str, int] = {}
            for ch in body["changes"]:
                reasons[ch["reason"]] = reasons.get(ch["reason"], 0) + 1
            cursors[who] = body["cursor"]
            out.append(f"{who:5} sync -> {len(body['changes'])} changes "
                       f"{json.dumps(reasons, sort_keys=True)}")
        elif op == "compact":
            n = app.state.store.compact(board, step["keep_last"])
            out.append(f"admin compact keep_last={step['keep_last']} -> dropped {n}")
        elif op == "show":
            out.append(f"      {step['column']:<6}: {column_order(step['column'])}")
        else:
            raise ValueError(op)

    # final convergence check: every replica equals the server
    server = {c["id"]: c for c in http.get(f"/api/boards/{board}").json()["cards"]}
    for who, rep in replicas.items():
        out.append(f"{who} replica matches server: {rep == server}")
    for name in scenario["columns"]:
        ordered = sorted((c for c in server.values() if col_name[c["column_id"]] == name),
                         key=lambda c: c["rank"])
        out.append(f"final {name:<6}: " + (" ".join(c["id"] for c in ordered) or "(empty)"))
    out.append(f"invariant problems: {app.state.store.check_invariants(board)}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=str(SAMPLES / "scenario.json"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    lines = run(json.loads(Path(args.scenario).read_text()))
    text = "\n".join(lines) + "\n"
    if args.check:
        expected = (SAMPLES / "scenario_expected.txt").read_text()
        if text != expected:
            print("transcript differs from samples/scenario_expected.txt", file=sys.stderr)
            return 1
        print("transcript matches samples/scenario_expected.txt")
        return 0
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
