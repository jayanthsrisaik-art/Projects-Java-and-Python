"""gitignore-style path patterns and a root-level .gitignore evaluator.

Semantics follow git's ``dir.c`` (``add_patterns_from_buffer``,
``parse_path_pattern``, ``match_basename``, ``match_pathname`` and the
parent-directory exclusion done by ``prep_exclude``):

* lines are split on ``\\n``; a trailing ``\\r`` is dropped;
* a line whose first byte is ``#`` is a comment (``\\#`` escapes it);
* trailing spaces are removed unless escaped with a backslash;
* a leading ``!`` negates (``\\!`` is a literal ``!``);
* one trailing ``/`` means "directories only" and is removed;
* a pattern with no remaining ``/`` matches the basename at any depth,
  otherwise it is anchored to the root (a leading ``/`` is dropped) and the
  whole relative path is matched with ``WM_PATHNAME``;
* the last matching pattern wins, and once a directory is excluded nothing
  below it can be re-included.

``prefix_quirk=True`` reproduces older git (observed in 2.43.0): match_pathname()
compared the literal prefix separately, so the ``**`` in ``a**/**`` counted as
segment-initial and the pattern matched the directory ``a``.  Current git
(observed in 2.56.0) no longer does this, so the default is ``False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .wildmatch import WildPattern, literal_prefix_len


def to_bytes(s: bytes | str) -> bytes:
    return s.encode("utf-8") if isinstance(s, str) else s


def trim_trailing_spaces(line: bytes) -> bytes:
    """Port of git's trim_trailing_spaces()."""
    last_space = None
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == 0x20:
            if last_space is None:
                last_space = i
        elif c == 0x5C:  # backslash: skip the escaped byte
            i += 1
            if i >= n:
                return line
            last_space = None
        else:
            last_space = None
        i += 1
    return line[:last_space] if last_space is not None else line


@dataclass
class PathPattern:
    """One parsed pattern (without the gitignore line handling)."""

    text: bytes  # the pattern after negation/trailing-slash processing
    negated: bool = False
    dir_only: bool = False
    basename_only: bool = False  # no slash: match basename at any depth
    raw: bytes = b""
    line_no: int = 0
    prefix_quirk: bool = False
    _wp: WildPattern = field(init=False, repr=False)

    def __post_init__(self) -> None:
        body = self.text
        if not self.basename_only and body.startswith(b"/"):
            body = body[1:]
        if self.basename_only:
            # match_basename(): whole pattern vs basename, no WM_PATHNAME.
            self._wp = WildPattern(body, pathname=False)
        else:
            # match_pathname(): WM_PATHNAME wildmatch on the path (see prefix_quirk).
            start = literal_prefix_len(body) if self.prefix_quirk else 0
            self._wp = WildPattern(body, pathname=True, segment_start=start)

    @classmethod
    def parse(cls, pattern: bytes | str, line_no: int = 0, allow_negation: bool = True,
              prefix_quirk: bool = False) -> "PathPattern":
        raw = to_bytes(pattern)
        p = raw
        negated = False
        if allow_negation and p.startswith(b"!"):
            negated = True
            p = p[1:]
        dir_only = False
        if p.endswith(b"/"):
            dir_only = True
            p = p[:-1]
        basename_only = b"/" not in p
        return cls(p, negated, dir_only, basename_only, raw, line_no, prefix_quirk)

    @property
    def valid(self) -> bool:
        return self._wp.valid

    @property
    def invalid_reason(self) -> str | None:
        return self._wp.reason

    def matches(self, path: bytes, is_dir: bool, ignore_case: bool = False) -> bool:
        """Does this pattern (ignoring negation) match ``path`` itself?"""
        if self.dir_only and not is_dir:
            return False
        if self.basename_only:
            target = path.rsplit(b"/", 1)[-1]
        else:
            target = path
        return self._wp.match(target, ignore_case)


def parse_gitignore(content: bytes | str, prefix_quirk: bool = False) -> list[PathPattern]:
    data = to_bytes(content)
    patterns: list[PathPattern] = []
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    for no, line in enumerate(lines, start=1):
        if line.endswith(b"\r"):
            line = line[:-1]
        if not line or line.startswith(b"#"):
            continue
        line = trim_trailing_spaces(line)
        patterns.append(PathPattern.parse(line, no, prefix_quirk=prefix_quirk))
    return patterns


@dataclass(frozen=True)
class IgnoreResult:
    ignored: bool
    pattern: PathPattern | None  # deciding pattern (may be a negation), like check-ignore -v


class GitIgnore:
    """Evaluate a single root-level .gitignore exactly like ``git check-ignore``."""

    def __init__(self, content: bytes | str, prefix_quirk: bool = False):
        self.patterns = parse_gitignore(content, prefix_quirk)
        self._dir_cache: dict[bytes, PathPattern | None] = {}

    def _last_match(self, path: bytes, is_dir: bool) -> PathPattern | None:
        for pat in reversed(self.patterns):
            if pat.matches(path, is_dir):
                return pat
        return None

    def _excluded_ancestor(self, path: bytes) -> PathPattern | None:
        parts = path.split(b"/")
        for k in range(1, len(parts)):
            prefix = b"/".join(parts[:k])
            if prefix not in self._dir_cache:
                self._dir_cache[prefix] = self._last_match(prefix, True)
            hit = self._dir_cache[prefix]
            if hit is not None and not hit.negated:
                return hit
        return None

    def check(self, path: bytes | str, is_dir: bool = False) -> IgnoreResult:
        p = to_bytes(path).strip(b"/")
        parent = self._excluded_ancestor(p)
        if parent is not None:
            return IgnoreResult(True, parent)
        hit = self._last_match(p, is_dir)
        return IgnoreResult(hit is not None and not hit.negated, hit)
