"""Translate git wildmatch patterns into byte regular expressions.

This is a behavioural port of git's ``wildmatch.c`` (used by .gitignore,
.gitattributes and pathspecs) rather than of ``fnmatch``.  The differences
that matter in practice:

* with ``WM_PATHNAME`` a single ``*``/``?``/``[...]`` never matches ``/``;
* ``**`` is only special when it is a whole path segment (``**/x``,
  ``x/**``, ``x/**/y``); anywhere else it behaves like ``*``;
* ``[...]`` follows git's parser exactly: ``!`` or ``^`` negates, a leading
  ``]`` is literal, ``\\`` escapes, reversed ranges match nothing, POSIX
  classes like ``[:alpha:]`` are supported and an unterminated bracket (or an
  unknown class) makes the *whole* pattern unmatchable;
* a trailing lone backslash makes the whole pattern unmatchable;
* matching is byte-wise (``?`` matches one byte of a UTF-8 sequence), as git does.

``translate`` returns the regex source, or ``None`` when git would never match
the pattern, together with a human-readable reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SLASH = ord("/")
STAR = ord("*")
BACKSLASH = ord("\\")

_UPPER = set(range(ord("A"), ord("Z") + 1))
_LOWER = set(range(ord("a"), ord("z") + 1))
_DIGIT = set(range(ord("0"), ord("9") + 1))
_PUNCT = {c for c in range(33, 127)} - _UPPER - _LOWER - _DIGIT
# C-locale ctype semantics, matching git's sane_ctype table.
CLASSES: dict[bytes, set[int]] = {
    b"alnum": _UPPER | _LOWER | _DIGIT,
    b"alpha": _UPPER | _LOWER,
    b"blank": {ord(" "), ord("\t")},
    b"cntrl": set(range(0, 32)) | {127},
    b"digit": _DIGIT,
    b"graph": set(range(33, 127)),
    b"lower": _LOWER,
    b"print": set(range(32, 127)),
    b"punct": _PUNCT,
    b"space": {ord(c) for c in " \t\n\v\f\r"},
    b"upper": _UPPER,
    b"xdigit": _DIGIT | set(b"abcdefABCDEF"),
}


class _Abort(Exception):
    """Git returns WM_ABORT_ALL: the pattern can never match anything."""


@dataclass(frozen=True)
class Translation:
    regex: bytes | None  # None => never matches
    reason: str | None = None  # why it never matches


def _byte_class(members: set[int], negated: bool) -> bytes:
    if not members:
        return b"(?!)" if not negated else b"[\\x00-\\xff]"
    items = sorted(members)
    parts: list[bytes] = []
    start = prev = items[0]
    for c in items[1:] + [None]:  # type: ignore[list-item]
        if c is not None and c == prev + 1:
            prev = c
            continue
        if start == prev:
            parts.append(b"\\x%02x" % start)
        else:
            parts.append(b"\\x%02x-\\x%02x" % (start, prev))
        if c is not None:
            start = prev = c
    return b"[" + (b"^" if negated else b"") + b"".join(parts) + b"]"


def _parse_bracket(pat: bytes, i: int) -> tuple[set[int], bool, int]:
    """Parse ``[...]`` starting at ``pat[i] == '['``; return (members, negated, next_index).

    Mirrors the do/while loop in wildmatch.c byte for byte.
    """
    n = len(pat)
    p = i + 1
    if p >= n:
        raise _Abort("unterminated '['")
    p_ch = pat[p]
    if p_ch == ord("^"):
        p_ch = ord("!")
    negated = p_ch == ord("!")
    if negated:
        p += 1
        if p >= n:
            raise _Abort("unterminated '['")
        p_ch = pat[p]
    prev = 0
    members: set[int] = set()
    while True:
        if p_ch == BACKSLASH:
            p += 1
            if p >= n:
                raise _Abort("unterminated '[' (dangling backslash)")
            p_ch = pat[p]
            members.add(p_ch)
        elif p_ch == ord("-") and prev and p + 1 < n and pat[p + 1] != ord("]"):
            p += 1
            p_ch = pat[p]
            if p_ch == BACKSLASH:
                p += 1
                if p >= n:
                    raise _Abort("unterminated '[' (dangling backslash)")
                p_ch = pat[p]
            members.update(range(prev, p_ch + 1))  # empty when reversed, like git
            p_ch = 0
        elif p_ch == ord("[") and p + 1 < n and pat[p + 1] == ord(":"):
            s = p + 2
            q = s
            while q < n and pat[q] != ord("]"):
                q += 1
            if q >= n:
                raise _Abort("unterminated '['")
            length = q - s - 1
            if length < 0 or pat[q - 1] != ord(":"):
                # No ":]" -> the '[' is an ordinary member; rescan from after it.
                p = s - 2
                p_ch = ord("[")
                members.add(p_ch)
            else:
                name = pat[s : s + length]
                if name not in CLASSES:
                    raise _Abort(f"unknown character class [:{name.decode('latin-1')}:]")
                members.update(CLASSES[name])
                p = q
                p_ch = 0
        else:
            members.add(p_ch)
        prev = p_ch
        p += 1
        if p >= n:
            raise _Abort("unterminated '['")
        p_ch = pat[p]
        if p_ch == ord("]"):
            return members, negated, p + 1


def literal_prefix_len(pat: bytes) -> int:
    """git's simple_length(): bytes before the first ``*?[\\``."""
    for i, c in enumerate(pat):
        if c in b"*?[\\":
            return i
    return len(pat)


def translate(pattern: bytes | str, pathname: bool = True, segment_start: int = 0) -> Translation:
    """Translate one wildmatch pattern to a regex usable with ``re.fullmatch``.

    ``pathname=True`` corresponds to git's ``WM_PATHNAME`` flag.
    ``segment_start`` reproduces a quirk of older git's ``match_pathname``
    (seen in 2.43.0, gone in 2.56.0): it compared the literal prefix with
    ``strncmp`` and handed only the rest to wildmatch, so a ``**`` directly
    after that prefix counted as segment-initial (``a**/**`` matched ``a``).
    """
    pat = pattern.encode("utf-8") if isinstance(pattern, str) else pattern
    out: list[bytes] = []
    n = len(pat)
    i = 0
    try:
        while i < n:
            c = pat[i]
            if c == STAR:
                j = i
                while j < n and pat[j] == STAR:
                    j += 1
                if j - i >= 2 and pathname:
                    prev_ok = i == 0 or i == segment_start or pat[i - 1] == SLASH
                    if prev_ok and j < n and pat[j] == SLASH:
                        out.append(b"(?:.*/)?")  # zero or more whole directories
                        i = j + 1
                        continue
                    if prev_ok and (j == n or pat[j : j + 2] == b"\\/"):
                        out.append(b".*")
                        i = j
                        continue
                out.append(b"[^/]*" if pathname else b".*")
                i = j
            elif c == ord("?"):
                out.append(b"[^/]" if pathname else b".")
                i += 1
            elif c == ord("["):
                members, negated, i = _parse_bracket(pat, i)
                if pathname:
                    members = set(members)
                    if negated:
                        members.add(SLASH)
                    else:
                        members.discard(SLASH)
                out.append(_byte_class(members, negated))
            elif c == BACKSLASH:
                if i + 1 >= n:
                    raise _Abort("trailing backslash")
                out.append(re.escape(pat[i + 1 : i + 2]))
                i += 2
            else:
                out.append(re.escape(pat[i : i + 1]))
                i += 1
    except _Abort as exc:
        return Translation(None, str(exc))
    return Translation(b"".join(out))


class WildPattern:
    """A compiled wildmatch pattern."""

    __slots__ = ("source", "reason", "_rx", "_rx_ci")

    def __init__(self, pattern: bytes | str, pathname: bool = True, segment_start: int = 0):
        self.source = pattern.encode("utf-8") if isinstance(pattern, str) else pattern
        t = translate(self.source, pathname, segment_start)
        self.reason = t.reason
        self._rx = re.compile(t.regex, re.DOTALL) if t.regex is not None else None
        self._rx_ci = (
            re.compile(t.regex, re.DOTALL | re.IGNORECASE) if t.regex is not None else None
        )

    @property
    def valid(self) -> bool:
        return self._rx is not None

    def match(self, text: bytes, ignore_case: bool = False) -> bool:
        rx = self._rx_ci if ignore_case else self._rx
        return rx is not None and rx.fullmatch(text) is not None


def wildmatch(pattern: bytes | str, text: bytes | str, pathname: bool = True) -> bool:
    t = text.encode("utf-8") if isinstance(text, str) else text
    return WildPattern(pattern, pathname).match(t)
