"""Command line interface: ``python -m ownermap <command>``."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from . import lint as lint_mod
from . import review as review_mod
from .codeowners import CodeOwners, Resolution
from .pathpattern import GitIgnore
from .registry import Registry


def read_list(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]


def load_files(args: argparse.Namespace) -> list[str] | None:
    if getattr(args, "files", None):
        return read_list(args.files)
    if getattr(args, "git", None):
        out = subprocess.run(["git", "-C", args.git, "ls-files", "-z"], check=True, capture_output=True).stdout
        return [p.decode("utf-8") for p in out.split(b"\0") if p]
    if getattr(args, "root", None):
        found = []
        for dirpath, dirnames, filenames in os.walk(args.root):
            dirnames[:] = sorted(d for d in dirnames if d != ".git")
            for f in sorted(filenames):
                found.append(os.path.relpath(os.path.join(dirpath, f), args.root).replace(os.sep, "/"))
        return found
    return None


def _tree_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--files", help="text file with one repository path per line")
    g.add_argument("--git", metavar="REPO", help="use `git ls-files` of this repository")
    g.add_argument("--root", help="walk this directory")


def cmd_who(args: argparse.Namespace) -> int:
    co = CodeOwners.from_file(args.codeowners)
    out = []
    for path in args.paths:
        rule = co.deciding_rule(path)
        entry = {"path": path, "owners": list(rule.owners) if rule else [],
                 "line": rule.line if rule else None, "pattern": rule.pattern if rule else None}
        if args.why:
            entry["matching_lines"] = [r.line for r in co.matching_rules(path)]
        out.append(entry)
    if args.format == "json":
        print(json.dumps(out, indent=2))
    else:
        for e in out:
            owners = " ".join(e["owners"]) or "(unowned)"
            where = f"line {e['line']}: {e['pattern']}" if e["line"] else "no matching rule"
            extra = f"  matched lines {e['matching_lines']}" if args.why else ""
            print(f"{e['path']}: {owners}  [{where}]{extra}")
    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    co = CodeOwners.from_file(args.codeowners)
    reg = Registry.from_file(args.registry) if args.registry else None
    report = lint_mod.lint(co, load_files(args), reg)
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(lint_mod.format_text(report, args.codeowners))
    if report.errors or (args.strict and report.warnings):
        return 1
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    co = CodeOwners.from_file(args.codeowners)
    files = load_files(args)
    if files is None:
        print("coverage needs --files, --git or --root", file=sys.stderr)
        return 2
    res = Resolution.build(co, files)
    by_dir: dict[str, list[int]] = {}
    for f in files:
        parts = f.split("/")
        d = "/".join(parts[: args.depth]) + "/" if len(parts) > args.depth else "(root)"
        rule = res.owner_of[f]
        row = by_dir.setdefault(d, [0, 0])
        row[0] += 1
        row[1] += bool(rule and rule.owners)
    rows = sorted(by_dir.items())
    total, owned = len(files), sum(o for _, (_, o) in rows)
    if args.format == "json":
        print(json.dumps([{"dir": d, "files": t, "owned": o} for d, (t, o) in rows], indent=2))
    else:
        width = max(len(d) for d, _ in rows)
        for d, (t, o) in rows:
            print(f"{d:<{width}}  {o:>4}/{t:<4} {100.0 * o / t:5.1f}%")
        print(f"{'TOTAL':<{width}}  {owned:>4}/{total:<4} {100.0 * owned / max(total, 1):5.1f}%")
    pct = 100.0 * owned / total if total else 100.0
    return 0 if args.min_coverage is None or pct >= args.min_coverage else 1


def cmd_review(args: argparse.Namespace) -> int:
    co = CodeOwners.from_file(args.codeowners)
    reg = Registry.from_file(args.registry) if args.registry else None
    changed = read_list(args.changed) if args.changed else []
    changed += args.paths
    plan = review_mod.plan_review(co, changed, reg, args.author)
    if args.format == "json":
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        print(review_mod.format_text(plan))
    return 1 if any(f.status == "unsatisfiable" for f in plan.files) else 0


def cmd_check_ignore(args: argparse.Namespace) -> int:
    with open(args.gitignore, "rb") as fh:
        gi = GitIgnore(fh.read())
    any_ignored = False
    for q in args.paths:
        res = gi.check(q.rstrip("/"), is_dir=q.endswith("/"))
        any_ignored |= res.ignored
        if args.verbose:
            src = f"{args.gitignore}:{res.pattern.line_no}:{res.pattern.raw.decode('utf-8', 'replace')}" if res.pattern else "::"
            print(f"{src}\t{q}")
        elif res.ignored:
            print(q)
    return 0 if any_ignored else 1


def cmd_bench(args: argparse.Namespace) -> int:
    from . import bench

    if args.what == "ignore":
        result = bench.bench_ignore(args.cases or 300, args.seed if args.seed is not None else 7)
    else:
        result = bench.bench_review(args.cases or 500, args.seed if args.seed is not None else 11)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ownermap", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("who", help="who owns these paths, and which line decided")
    p.add_argument("-c", "--codeowners", required=True)
    p.add_argument("paths", nargs="+")
    p.add_argument("--why", action="store_true", help="also list every matching line")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.set_defaults(func=cmd_who)

    p = sub.add_parser("lint", help="find broken, dead and shadowed rules")
    p.add_argument("-c", "--codeowners", required=True)
    _tree_args(p)
    p.add_argument("--registry", help="JSON with users/teams/emails")
    p.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.set_defaults(func=cmd_lint)

    p = sub.add_parser("coverage", help="owned/total files per directory")
    p.add_argument("-c", "--codeowners", required=True)
    _tree_args(p)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--min-coverage", type=float, help="exit 1 if total coverage %% is below this")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("review", help="smallest reviewer set for a change")
    p.add_argument("-c", "--codeowners", required=True)
    p.add_argument("--changed", help="file with changed paths, one per line")
    p.add_argument("paths", nargs="*")
    p.add_argument("--registry")
    p.add_argument("--author", help="PR author; their approval does not count")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("check-ignore", help="git check-ignore for a root .gitignore (dirs end with /)")
    p.add_argument("--gitignore", required=True)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_check_ignore)

    p = sub.add_parser("bench", help="measured benchmarks")
    p.add_argument("what", choices=["ignore", "review"])
    p.add_argument("--cases", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--out")
    p.set_defaults(func=cmd_bench)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
