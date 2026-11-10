"""CODEOWNERS parsing and ownership resolution.

Rules follow GitHub's documented behaviour:

* ``#`` starts a comment, both on its own line and after the owners;
* each rule is ``<pattern> <owner>...``; a rule with no owners explicitly
  leaves matching files without an owner;
* owners are ``@user``, ``@org/team`` or an e-mail address;
* the **last** matching rule wins;
* patterns use gitignore syntax, except that ``!`` negation, ``[ ]`` ranges
  and ``\\#`` escapes are not supported — such lines are reported as errors
  and skipped (GitHub skips lines with invalid syntax);
* a pattern that matches a directory owns everything beneath it
  (``apps/``, ``/build/logs/``, ``**/logs``, ``/apps/github``), except that a
  pattern ending in ``*`` only owns what it matches directly — GitHub's docs
  state that ``docs/*`` owns ``docs/getting-started.md`` but not
  ``docs/build-app/troubleshooting.md``.

Paths with spaces are written with backslash-escaped spaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .pathpattern import PathPattern, to_bytes

USER_RE = re.compile(r"^@[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")
TEAM_RE = re.compile(r"^@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?/[A-Za-z0-9][A-Za-z0-9._-]*$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def owner_kind(token: str) -> str | None:
    if TEAM_RE.match(token):
        return "team"
    if USER_RE.match(token):
        return "user"
    if EMAIL_RE.match(token):
        return "email"
    return None


@dataclass(frozen=True)
class Problem:
    line: int
    code: str
    message: str
    severity: str = "error"  # error | warning | info


@dataclass
class Rule:
    line: int
    pattern: str
    owners: tuple[str, ...]
    matcher: PathPattern = field(repr=False)
    descendants: bool = True  # does a matched directory own its contents?

    def matches_self(self, path: bytes, is_dir: bool, ignore_case: bool = False) -> bool:
        return self.matcher.matches(path, is_dir, ignore_case)


def _tokenize(line: str) -> list[str]:
    """Split on unescaped whitespace; ``\\ `` stays inside a token."""
    tokens: list[str] = []
    cur: list[str] = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line) and line[i + 1] in " \t":
            cur.append(c + line[i + 1])
            i += 2
            continue
        if c in " \t":
            if cur:
                tokens.append("".join(cur))
                cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        tokens.append("".join(cur))
    return tokens


def _ends_with_unescaped_star(pattern: str) -> bool:
    if not pattern.endswith("*"):
        return False
    backslashes = len(pattern[:-1]) - len(pattern[:-1].rstrip("\\"))
    return backslashes % 2 == 0


@dataclass
class CodeOwners:
    rules: list[Rule]
    problems: list[Problem]

    @classmethod
    def parse(cls, text: str) -> "CodeOwners":
        rules: list[Rule] = []
        problems: list[Problem] = []
        for no, raw in enumerate(text.splitlines(), start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("\\#"):
                problems.append(Problem(no, "unsupported-syntax",
                                        "escaping a leading '#' with '\\' is not supported by GitHub; line skipped"))
                continue
            if stripped.startswith("#"):
                continue
            tokens = _tokenize(stripped)
            pattern, rest = tokens[0], tokens[1:]
            owners: list[str] = []
            for tok in rest:
                if tok.startswith("#"):
                    break  # inline comment
                owners.append(tok)
            bad = False
            if pattern.startswith("!"):
                problems.append(Problem(no, "unsupported-syntax",
                                        f"negation ('{pattern}') is not supported in CODEOWNERS; line skipped"))
                bad = True
            if "[" in pattern:
                problems.append(Problem(no, "unsupported-syntax",
                                        f"character ranges ('{pattern}') are not supported in CODEOWNERS; line skipped"))
                bad = True
            for o in owners:
                if owner_kind(o) is None:
                    problems.append(Problem(no, "invalid-owner",
                                            f"'{o}' is not a @user, @org/team or e-mail; line skipped"))
                    bad = True
            if bad:
                continue
            # "\ " is already a literal space in wildmatch syntax.
            matcher = PathPattern.parse(pattern, no, allow_negation=False)
            if not matcher.valid:
                problems.append(Problem(no, "never-matches",
                                        f"pattern '{pattern}' can never match ({matcher.invalid_reason}); line skipped"))
                continue
            rules.append(Rule(no, pattern, tuple(owners), matcher, not _ends_with_unescaped_star(pattern)))
        return cls(rules, problems)

    @classmethod
    def from_file(cls, path: str) -> "CodeOwners":
        with open(path, encoding="utf-8") as fh:
            return cls.parse(fh.read())

    # -- matching ---------------------------------------------------------
    def rule_matches(self, rule: Rule, path: str | bytes, ignore_case: bool = False) -> bool:
        p = to_bytes(path).strip(b"/")
        if rule.matches_self(p, False, ignore_case):
            return True
        if rule.descendants:
            parts = p.split(b"/")
            for k in range(1, len(parts)):
                if rule.matches_self(b"/".join(parts[:k]), True, ignore_case):
                    return True
        return False

    def deciding_rule(self, path: str | bytes) -> Rule | None:
        for rule in reversed(self.rules):
            if self.rule_matches(rule, path):
                return rule
        return None

    def owners_of(self, path: str | bytes) -> tuple[str, ...]:
        rule = self.deciding_rule(path)
        return rule.owners if rule else ()

    def matching_rules(self, path: str | bytes) -> list[Rule]:
        """Every rule matching ``path``, in file order (the last one decides)."""
        return [r for r in self.rules if self.rule_matches(r, path)]


@dataclass
class Resolution:
    """Ownership of a whole file tree, computed once."""

    codeowners: CodeOwners
    files: list[str]
    matches: dict[int, list[str]]  # rule line -> files it matches
    decided: dict[int, list[str]]  # rule line -> files it decides
    owner_of: dict[str, Rule | None]

    @classmethod
    def build(cls, co: CodeOwners, files: list[str]) -> "Resolution":
        matches: dict[int, list[str]] = {r.line: [] for r in co.rules}
        decided: dict[int, list[str]] = {r.line: [] for r in co.rules}
        owner_of: dict[str, Rule | None] = {}
        # Cache directory verdicts: many files share ancestors.
        dir_cache: dict[tuple[int, bytes], bool] = {}
        for f in files:
            p = to_bytes(f).strip(b"/")
            parts = p.split(b"/")
            last: Rule | None = None
            for r in co.rules:
                hit = r.matches_self(p, False)
                if not hit and r.descendants:
                    for k in range(1, len(parts)):
                        key = (r.line, b"/".join(parts[:k]))
                        if key not in dir_cache:
                            dir_cache[key] = r.matches_self(key[1], True)
                        if dir_cache[key]:
                            hit = True
                            break
                if hit:
                    matches[r.line].append(f)
                    last = r
            owner_of[f] = last
            if last is not None:
                decided[last.line].append(f)
        return cls(co, files, matches, decided, owner_of)
