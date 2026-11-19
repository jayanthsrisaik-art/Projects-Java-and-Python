"""Keyed pseudonyms and per-subject date shifting.

Tokens are HMAC-SHA256(key, domain, value): stable across tables and runs for
the same key (so joins survive), unlinkable without the key, and different
per domain (a patient id and an encounter id with the same digits get
unrelated tokens).  A plain unkeyed hash would be reversible by hashing every
plausible id.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import os
from pathlib import Path

MIN_KEY_BYTES = 16
DEMO_KEY = b"safeshare-demo-key-NOT-FOR-REAL-DATA"


class KeyError_(ValueError):
    pass


def load_key(key_file: str | None = None, env: str = "SAFESHARE_KEY",
             demo: bool = False) -> bytes:
    if demo:
        return DEMO_KEY
    if key_file:
        key = Path(key_file).read_bytes().strip()
    elif os.environ.get(env):
        key = os.environ[env].encode()
    else:
        raise KeyError_(f"no key: pass --key-file, set ${env}, or use --demo-key "
                        "for synthetic data")
    if len(key) < MIN_KEY_BYTES:
        raise KeyError_(f"key must be at least {MIN_KEY_BYTES} bytes")
    return key


class Pseudonymizer:
    def __init__(self, key: bytes, max_shift_days: int = 180):
        if len(key) < MIN_KEY_BYTES:
            raise KeyError_(f"key must be at least {MIN_KEY_BYTES} bytes")
        self._key = key
        self.max_shift = max_shift_days
        self._cache: dict = {}

    def _mac(self, *parts: str) -> bytes:
        msg = "\x00".join(parts).encode("utf-8")
        return hmac.new(self._key, msg, hashlib.sha256).digest()

    def token(self, domain: str, value: str) -> str:
        ck = ("t", domain, value)
        tok = self._cache.get(ck)
        if tok is None:
            raw = base64.b32encode(self._mac("token", domain, value)[:10]).decode()
            tok = f"{domain[:3]}_{raw.lower()}"
            self._cache[ck] = tok
        return tok

    def shift_days(self, domain: str, subject: str) -> int:
        """Per-subject offset in [-max, +max] \\ {0}: intervals within a subject
        are preserved exactly, absolute dates are not."""
        ck = ("s", domain, subject)
        off = self._cache.get(ck)
        if off is None:
            n = int.from_bytes(self._mac("shift", domain, subject)[:8], "big")
            off = n % (2 * self.max_shift) - self.max_shift
            if off >= 0:
                off += 1          # skip 0 so every subject really is shifted
            self._cache[ck] = off
        return off

    def shift_date(self, domain: str, subject: str, value: str) -> str:
        value = value.strip()
        if not value:
            return ""
        d = dt.date.fromisoformat(value[:10])
        shifted = d + dt.timedelta(days=self.shift_days(domain, subject))
        return shifted.isoformat() + value[10:]   # keep any time-of-day suffix
