"""Differential testing of the .gitignore evaluator against real ``git check-ignore``.

Each case builds a throwaway repository with a random tree and a random
.gitignore, asks git about every file *and* directory, and compares the
ignored/not-ignored verdict plus the deciding line number with ours.

A naive ``fnmatch``-based evaluator (the approach many tools take) is run on
the same cases as a baseline.
"""

from __future__ import annotations

import fnmatch
import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from .pathpattern import GitIgnore

NAMES = ["a", "b", "ab", "ba", "a.b", "b.c", "x y", "[a]", "!n", "#h", "A", "é", "aa", "logs"]
SEGMENTS = [
    "a", "b", "ab", "a.b", "logs", "A", "é", "x y", "[a]", "!n", "#h",
    "*", "?", "??", "a*", "*b", "*.b", "*.c", "[ab]", "[!a]", "[^a]", "[a-c]", "[c-a]",
    "[]a]", "[[:alpha:]]", "[[:digit:]x]", "[[:bogus:]]", "[[:]", "\\*", "\\[a]",
    "**", "a**", "**b", "[", "x\\ y", "?.?", "[a-]", "[-b]", "\\#h", "\\!n", "é?",
]


def git_available() -> bool:
    return shutil.which("git") is not None


_QUIRK: bool | None = None


def git_has_prefix_quirk() -> bool:
    """Ask the installed git whether ``a**/**`` ignores the directory ``a`` (older git does)."""
    global _QUIRK
    if _QUIRK is None:
        _QUIRK = git_check_ignore("a**/**\n", ["a/b"], ["a"])["a"][0]
    return _QUIRK


def random_tree(rng: random.Random, n_files: int) -> tuple[list[str], list[str]]:
    files: set[str] = set()
    while len(files) < n_files:
        depth = rng.choice([1, 1, 2, 2, 3, 4])
        files.add("/".join(rng.choice(NAMES) for _ in range(depth)))
    # A file and a directory cannot share a path.
    files = {f for f in files if not any(g.startswith(f + "/") for g in files)}
    dirs = sorted({"/".join(f.split("/")[:k]) for f in files for k in range(1, f.count("/") + 1)})
    return sorted(files), dirs


def random_pattern(rng: random.Random) -> str:
    segs = [rng.choice(SEGMENTS) for _ in range(rng.choice([1, 1, 1, 2, 2, 3]))]
    p = "/".join(segs)
    r = rng.random()
    if r < 0.2:
        p = "/" + p
    elif r < 0.3:
        p = "**/" + p
    if rng.random() < 0.1:
        p += "/**"
    if rng.random() < 0.25:
        p += "/"
    if rng.random() < 0.25:
        p = "!" + p
    r = rng.random()
    if r < 0.08:
        p += "  "
    elif r < 0.12:
        p += "\\ "
    return p


def random_gitignore(rng: random.Random) -> str:
    lines = []
    for _ in range(rng.randint(2, 7)):
        r = rng.random()
        if r < 0.05:
            lines.append("# comment")
        elif r < 0.08:
            lines.append("")
        else:
            lines.append(random_pattern(rng))
    return "\n".join(lines) + "\n"


def git_check_ignore(gitignore: str, files: list[str], queries: list[str]) -> dict[str, tuple[bool, int]]:
    """Return {path: (ignored, deciding_line or 0)} as reported by git."""
    with tempfile.TemporaryDirectory(prefix="ownermap-git-") as tmp:
        subprocess.run(["git", "init", "-q", tmp], check=True)
        for f in files:
            full = os.path.join(tmp, f)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            open(full, "wb").close()
        with open(os.path.join(tmp, ".gitignore"), "wb") as fh:
            fh.write(gitignore.encode("utf-8"))
        env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", HOME=tmp, GIT_CONFIG_GLOBAL=os.devnull)
        proc = subprocess.run(
            ["git", "-c", "core.ignorecase=false", "-c", "core.quotepath=false",
             "check-ignore", "--no-index", "-v", "-n", "-z", "--stdin"],
            cwd=tmp, input=b"\0".join(q.encode("utf-8") for q in queries) + b"\0",
            capture_output=True, env=env,
        )
        if proc.returncode not in (0, 1):
            raise RuntimeError(proc.stderr.decode("utf-8", "replace"))
    fields = proc.stdout.split(b"\0")
    out: dict[str, tuple[bool, int]] = {}
    for k in range(0, len(fields) - 3, 4):
        _source, line, pattern, path = fields[k : k + 4]
        line_no = int(line) if line else 0
        ignored = bool(line) and not pattern.startswith(b"!")
        out[path.decode("utf-8")] = (ignored, line_no)
    return out


def naive_check(gitignore: str, path: str, is_dir: bool) -> tuple[bool, int]:
    """A typical fnmatch-based implementation: last match wins, no parent exclusion."""
    verdict = (False, 0)
    for no, line in enumerate(gitignore.split("\n"), start=1):
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        neg = line.startswith("!")
        pat = line[1:] if neg else line
        dir_only = pat.endswith("/")
        pat = pat.rstrip("/")
        if dir_only and not is_dir:
            continue
        target = path if "/" in pat else path.rsplit("/", 1)[-1]
        if fnmatch.fnmatchcase(target, pat.lstrip("/")):
            verdict = (not neg, no)
    return verdict


@dataclass
class DiffStats:
    cases: int = 0
    queries: int = 0
    ours_mismatch: int = 0
    naive_mismatch: int = 0
    ours_line_mismatch: int = 0
    cases_naive_wrong: int = 0
    first_mismatch: dict | None = None


def run_differential(cases: int = 200, seed: int = 7, files_per_case: int = 18,
                     prefix_quirk: bool | None = None) -> DiffStats:
    """Compare with the installed git; ``prefix_quirk=None`` probes git for its behaviour."""
    if prefix_quirk is None:
        prefix_quirk = git_has_prefix_quirk()
    rng = random.Random(seed)
    stats = DiffStats()
    for _ in range(cases):
        files, dirs = random_tree(rng, files_per_case)
        gi = random_gitignore(rng)
        queries = files + dirs
        truth = git_check_ignore(gi, files, queries)
        ours = GitIgnore(gi, prefix_quirk)
        dirset = set(dirs)
        stats.cases += 1
        naive_wrong_here = False
        for q in queries:
            if q not in truth:
                raise RuntimeError(f"git gave no answer for {q!r}")
            g_ign, g_line = truth[q]
            res = ours.check(q, q in dirset)
            o_line = res.pattern.line_no if res.pattern else 0
            stats.queries += 1
            if res.ignored != g_ign:
                stats.ours_mismatch += 1
                if stats.first_mismatch is None:
                    stats.first_mismatch = {"gitignore": gi, "path": q, "git": truth[q], "ours": (res.ignored, o_line)}
            elif o_line != g_line:
                stats.ours_line_mismatch += 1
                if stats.first_mismatch is None:
                    stats.first_mismatch = {"gitignore": gi, "path": q, "git": truth[q], "ours": (res.ignored, o_line)}
            if naive_check(gi, q, q in dirset)[0] != g_ign:
                stats.naive_mismatch += 1
                naive_wrong_here = True
        stats.cases_naive_wrong += naive_wrong_here
    return stats
