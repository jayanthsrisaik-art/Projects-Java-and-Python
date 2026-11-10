"""CODEOWNERS linting against a concrete file tree and an optional owner registry."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .codeowners import CodeOwners, Problem, Resolution
from .registry import Registry

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass
class LintReport:
    problems: list[Problem]
    total_files: int
    owned_files: int
    explicitly_unowned: int
    unowned_dirs: list[tuple[str, int]]

    @property
    def errors(self) -> int:
        return sum(p.severity == "error" for p in self.problems)

    @property
    def warnings(self) -> int:
        return sum(p.severity == "warning" for p in self.problems)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "errors": self.errors,
                "warnings": self.warnings,
                "files": self.total_files,
                "owned": self.owned_files,
                "explicitly_unowned": self.explicitly_unowned,
                "coverage_pct": round(100.0 * self.owned_files / self.total_files, 1) if self.total_files else None,
            },
            "problems": [p.__dict__ for p in self.problems],
            "unowned_dirs": [{"dir": d, "files": n} for d, n in self.unowned_dirs],
        }


def _top_dir(path: str, depth: int = 1) -> str:
    parts = path.split("/")
    return "/".join(parts[:depth]) + "/" if len(parts) > depth else "(root)"


def lint(co: CodeOwners, files: list[str] | None = None, registry: Registry | None = None) -> LintReport:
    problems = list(co.problems)
    seen: dict[str, int] = {}
    for r in co.rules:
        if r.pattern in seen:
            problems.append(Problem(r.line, "duplicate-pattern",
                                    f"pattern '{r.pattern}' already appears on line {seen[r.pattern]}; "
                                    "the earlier rule can never decide ownership", "warning"))
        seen.setdefault(r.pattern, r.line)
        if not r.owners:
            problems.append(Problem(r.line, "no-owners",
                                    f"'{r.pattern}' has no owners: matching files are explicitly unowned", "info"))

    if registry is not None:
        for r in co.rules:
            for o in r.owners:
                if not registry.knows(o):
                    problems.append(Problem(r.line, "unknown-owner", f"owner '{o}' is not in the registry", "error"))
                elif registry.is_team(o) and not registry.members(o):
                    problems.append(Problem(r.line, "empty-team",
                                            f"team '{o}' has no members, nobody can approve for it", "warning"))

    total = owned = explicit = 0
    unowned_dirs: list[tuple[str, int]] = []
    if files is not None:
        res = Resolution.build(co, files)
        for r in co.rules:
            matched, decided = res.matches[r.line], res.decided[r.line]
            if not matched:
                hint = ""
                if any(co.rule_matches(r, f, ignore_case=True) for f in files):
                    hint = " (it would match if case were ignored: CODEOWNERS paths are case-sensitive)"
                problems.append(Problem(r.line, "no-match", f"'{r.pattern}' matches no file in the tree{hint}", "warning"))
            elif not decided:
                takers = Counter(res.owner_of[f].line for f in matched)  # type: ignore[union-attr]
                lines = ", ".join(str(n) for n, _ in takers.most_common(3))
                problems.append(Problem(r.line, "shadowed",
                                        f"'{r.pattern}' matches {len(matched)} file(s) but later rules "
                                        f"(line {lines}) override all of them", "warning"))
        total = len(files)
        unowned = Counter()
        for f in files:
            rule = res.owner_of[f]
            if rule is not None and rule.owners:
                owned += 1
            else:
                if rule is not None:
                    explicit += 1
                unowned[_top_dir(f)] += 1
        unowned_dirs = sorted(unowned.items(), key=lambda kv: (-kv[1], kv[0]))
        not_explicit = total - owned - explicit
        if not_explicit:
            problems.append(Problem(0, "unowned-files",
                                    f"{not_explicit} file(s) match no rule (top: "
                                    + ", ".join(f"{d} {n}" for d, n in unowned_dirs[:3]) + ")", "info"))
    problems.sort(key=lambda p: (p.line, SEVERITY_ORDER[p.severity], p.code))
    return LintReport(problems, total, owned, explicit, unowned_dirs)


def format_text(report: LintReport, path: str = "CODEOWNERS") -> str:
    out = []
    for p in report.problems:
        loc = f"{path}:{p.line}" if p.line else path
        out.append(f"{loc}: {p.severity}: [{p.code}] {p.message}")
    s = report.to_dict()["summary"]
    cov = f", coverage {s['coverage_pct']}% ({s['owned']}/{s['files']} files owned)" if s["files"] else ""
    out.append(f"{s['errors']} error(s), {s['warnings']} warning(s){cov}")
    return "\n".join(out)
