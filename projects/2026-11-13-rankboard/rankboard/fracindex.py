"""Fractional indexing: order keys you can always insert between.

A key is ``<integer part><fraction part>`` over the base-62 alphabet ``0-9A-Za-z``
(which is also ASCII order, so plain string comparison == key order, in Python,
JavaScript and SQLite's default BINARY collation alike).

* The first character of the integer part encodes its length: ``a``..``z`` mean
  2..27 characters (non-negative integers), ``Z``..``A`` mean 2..27 characters
  (negative integers). ``a0`` is the first key ever handed out.
* Appending after the last key increments the integer part, so a column that
  only ever grows at the bottom keeps keys of length ~log62(n) instead of
  growing by one character every ~6 appends (what a fraction-only scheme does).
* Inserting between two keys with the same integer part takes the midpoint of
  their fraction parts, which never ends in ``0`` so there is always room below.

The same algorithm is implemented in ``web/src/fracindex.ts``; both are checked
against ``samples/key_vectors.json`` so client-side optimistic placement predicts
exactly the key the server will compute.
"""
from __future__ import annotations

from typing import Optional

DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BASE = len(DIGITS)
_INDEX = {c: i for i, c in enumerate(DIGITS)}
SMALLEST_INTEGER = "A" + "0" * 26


class KeyError_(ValueError):
    """Raised for malformed keys or impossible requests (a >= b)."""


def _midpoint(a: str, b: Optional[str]) -> str:
    """A fraction string strictly between fractions ``a`` and ``b`` (b=None means 1)."""
    if b is not None and a >= b:
        raise KeyError_(f"midpoint: {a!r} >= {b!r}")
    if a.endswith("0") or (b is not None and b.endswith("0")):
        raise KeyError_("midpoint: trailing zero")
    if b is not None:
        n = 0
        while (a[n] if n < len(a) else "0") == b[n]:
            n += 1
        if n > 0:
            return b[:n] + _midpoint(a[n:], b[n:])
    digit_a = _INDEX[a[0]] if a else 0
    digit_b = _INDEX[b[0]] if b is not None else BASE
    if digit_b - digit_a > 1:
        # (x + y + 1) // 2 == JavaScript Math.round(0.5 * (x + y)) for ints >= 0
        return DIGITS[(digit_a + digit_b + 1) // 2]
    if b is not None and len(b) > 1:
        return b[:1]
    return DIGITS[digit_a] + _midpoint(a[1:], None)


def _integer_length(head: str) -> int:
    if "a" <= head <= "z":
        return ord(head) - ord("a") + 2
    if "A" <= head <= "Z":
        return ord("Z") - ord(head) + 2
    raise KeyError_(f"invalid key head {head!r}")


def _integer_part(key: str) -> str:
    n = _integer_length(key[0])
    if n > len(key):
        raise KeyError_(f"key too short for its head: {key!r}")
    return key[:n]


def validate_key(key: str) -> None:
    if not isinstance(key, str) or not key:
        raise KeyError_("key must be a non-empty string")
    if any(c not in _INDEX for c in key):
        raise KeyError_(f"key has characters outside base62: {key!r}")
    if key == SMALLEST_INTEGER:
        raise KeyError_("key is the reserved smallest integer")
    integer = _integer_part(key)
    if key[len(integer):].endswith("0"):
        raise KeyError_(f"key fraction ends in 0: {key!r}")


def is_valid_key(key: str) -> bool:
    try:
        validate_key(key)
    except KeyError_:
        return False
    return True


def _increment_integer(x: str) -> Optional[str]:
    head, digs = x[0], list(x[1:])
    for i in range(len(digs) - 1, -1, -1):
        d = _INDEX[digs[i]] + 1
        if d == BASE:
            digs[i] = "0"
        else:
            digs[i] = DIGITS[d]
            return head + "".join(digs)
    # carried out of every digit: move to the next length class
    if head == "Z":
        return "a0"
    if head == "z":
        return None
    new_head = chr(ord(head) + 1)
    if new_head > "a":
        digs.append("0")
    else:
        digs.pop()
    return new_head + "".join(digs)


def _decrement_integer(x: str) -> Optional[str]:
    head, digs = x[0], list(x[1:])
    for i in range(len(digs) - 1, -1, -1):
        d = _INDEX[digs[i]] - 1
        if d == -1:
            digs[i] = "z"
        else:
            digs[i] = DIGITS[d]
            return head + "".join(digs)
    if head == "a":
        return "Zz"
    if head == "A":
        return None
    new_head = chr(ord(head) - 1)
    if new_head < "Z":
        digs.append("z")
    else:
        digs.pop()
    return new_head + "".join(digs)


def key_between(a: Optional[str], b: Optional[str]) -> str:
    """Return a key k with a < k < b. ``None`` means unbounded on that side."""
    if a is not None:
        validate_key(a)
    if b is not None:
        validate_key(b)
    if a is not None and b is not None and a >= b:
        raise KeyError_(f"key_between: {a!r} >= {b!r}")

    if a is None:
        if b is None:
            return "a0"
        ib = _integer_part(b)
        fb = b[len(ib):]
        if ib == SMALLEST_INTEGER:
            return ib + _midpoint("", fb)
        if ib < b:  # b has a fraction: its bare integer part sorts before it
            return ib
        res = _decrement_integer(ib)
        if res is None:
            raise KeyError_("cannot decrement any further")
        if res == SMALLEST_INTEGER:  # reserved as a bare key; give it a fraction
            return res + _midpoint("", None)
        return res

    ia = _integer_part(a)
    fa = a[len(ia):]
    if b is None:
        nxt = _increment_integer(ia)
        return ia + _midpoint(fa, None) if nxt is None else nxt

    ib = _integer_part(b)
    fb = b[len(ib):]
    if ia == ib:
        return ia + _midpoint(fa, fb)
    nxt = _increment_integer(ia)
    if nxt is None:
        raise KeyError_("cannot increment any further")
    if nxt < b:
        return nxt
    return ia + _midpoint(fa, None)


def keys_sequence(n: int) -> list[str]:
    """``n`` fresh, short, increasing keys (used to rebalance a column)."""
    out: list[str] = []
    prev: Optional[str] = None
    for _ in range(n):
        prev = key_between(prev, None)
        out.append(prev)
    return out


def fraction_only_between(a: Optional[str], b: Optional[str]) -> str:
    """Baseline without an integer part: keys are plain fractions in (0, 1).

    Used only by the benchmark to show why the integer part matters.
    """
    return _midpoint(a or "", b)
