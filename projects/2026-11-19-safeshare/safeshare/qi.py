"""Quasi-identifier kinds: encoding, generalised cells, formatting and parsing.

A *cell* is the released (generalised) value of one quasi-identifier for one
record.  Ordered kinds (date, numeric, zip) use an inclusive integer interval
``(lo, hi)``; categorical kinds use a ``frozenset`` of the original values.
Every released cell is formatted so it can be parsed back without the
original data - the attack and utility evaluators work only from the
released CSV text.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

ORDERED_KINDS = ("date", "numeric", "zip")
KINDS = ORDERED_KINDS + ("categorical",)

# Full-domain generalisation hierarchies (used by the lattice baseline).
DATE_LEVELS = ("day", "month", "year", "5y", "10y", "*")
ZIP_LEVELS = (5, 4, 3, 2, 0)            # digits kept
NUMERIC_BANDS = (1, 5, 10, 20, None)    # band width; None = fully suppressed
CATEGORICAL_LEVELS = ("exact", "*")


class QIError(ValueError):
    pass


@dataclass
class QISpec:
    """One quasi-identifier column and the global domain observed for it."""

    name: str
    kind: str
    lo: int = 0
    hi: int = 0
    domain: frozenset = field(default_factory=frozenset)
    zip_values: tuple = ()   # sorted distinct zips (public zip list stand-in)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise QIError(f"unknown QI kind {self.kind!r} for {self.name!r}")

    # ---------------------------------------------------------------- encode
    def encode(self, raw: str):
        raw = (raw or "").strip()
        if raw == "":
            raise QIError(f"missing value in quasi-identifier {self.name!r}")
        if self.kind == "date":
            try:
                return dt.date.fromisoformat(raw).toordinal()
            except ValueError as exc:
                raise QIError(f"{self.name}: bad ISO date {raw!r}") from exc
        if self.kind == "numeric":
            try:
                return int(raw)
            except ValueError as exc:
                raise QIError(f"{self.name}: not an integer: {raw!r}") from exc
        if self.kind == "zip":
            if len(raw) != 5 or not raw.isdigit():
                raise QIError(f"{self.name}: zip must be 5 digits, got {raw!r}")
            return int(raw)
        return raw

    def fit(self, values) -> "QISpec":
        values = list(values)
        if not values:
            raise QIError(f"no values for {self.name!r}")
        if self.kind == "categorical":
            self.domain = frozenset(values)
        else:
            self.lo, self.hi = min(values), max(values)
            if self.kind == "zip":
                self.zip_values = tuple(sorted(set(values)))
        return self

    # ------------------------------------------------------------------ cells
    def point(self, v):
        return frozenset([v]) if self.kind == "categorical" else (v, v)

    def contains(self, cell, v) -> bool:
        if self.kind == "categorical":
            return v in cell
        return cell[0] <= v <= cell[1]

    def ncp(self, cell) -> float:
        """Normalised certainty penalty of one cell (0 = exact, 1 = suppressed)."""
        if self.kind == "categorical":
            d = len(self.domain)
            return 0.0 if d <= 1 else (len(cell) - 1) / (d - 1)
        span = self.hi - self.lo
        if span <= 0:
            return 0.0
        return min(1.0, (min(cell[1], self.hi) - max(cell[0], self.lo)) / span)

    def padded_bounds(self) -> tuple[int, int]:
        """Outer range rounded to a coarse grid so the extreme records' exact
        values (the oldest patient's birth date) are not range endpoints."""
        if self.kind == "date":
            a, b = dt.date.fromordinal(self.lo), dt.date.fromordinal(self.hi)
            return (dt.date(a.year, 1, 1).toordinal(), dt.date(b.year, 12, 31).toordinal())
        step = 100 if self.kind == "zip" else 10
        return ((self.lo // step) * step, (self.hi // step) * step + step - 1)

    # -------------------------------------------------------------- hierarchy
    def levels(self) -> int:
        return {"date": len(DATE_LEVELS), "zip": len(ZIP_LEVELS),
                "numeric": len(NUMERIC_BANDS),
                "categorical": len(CATEGORICAL_LEVELS)}[self.kind]

    def generalise(self, v, level: int):
        """Full-domain generalisation of value ``v`` to hierarchy ``level``."""
        if level == 0:
            return self.point(v)
        if self.kind == "categorical":
            return self.domain
        if self.kind == "zip":
            digits = ZIP_LEVELS[level]
            block = 10 ** (5 - digits)
            lo = (v // block) * block
            return (lo, lo + block - 1)
        if level == self.levels() - 1:
            # fully suppressed: the hull of the coarsest blocks, so levels nest
            return (self.generalise(self.lo, level - 1)[0],
                    self.generalise(self.hi, level - 1)[1])
        if self.kind == "numeric":
            width = NUMERIC_BANDS[level]
            lo = (v // width) * width
            return (lo, lo + width - 1)
        d = dt.date.fromordinal(v)
        name = DATE_LEVELS[level]
        if name == "month":
            lo = d.replace(day=1)
            hi = (lo.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
            return (lo.toordinal(), hi.toordinal() - 1)
        if name in ("year", "5y", "10y"):
            width = {"year": 1, "5y": 5, "10y": 10}[name]
            y0 = (d.year // width) * width
            return (dt.date(y0, 1, 1).toordinal(),
                    dt.date(y0 + width, 1, 1).toordinal() - 1)
        raise QIError(f"unknown date level {name!r}")

    # ------------------------------------------------------------- formatting
    def format(self, cell) -> str:
        if self.kind == "categorical":
            return "|".join(sorted(cell))
        lo, hi = cell
        if self.kind == "date":
            a, b = dt.date.fromordinal(lo), dt.date.fromordinal(hi)
            return a.isoformat() if lo == hi else f"{a.isoformat()}..{b.isoformat()}"
        if self.kind == "zip":
            for digits in (4, 3, 2, 1, 0):
                block = 10 ** (5 - digits)
                if lo % block == 0 and hi == lo + block - 1:
                    return f"{lo:05d}"[:digits] + "*" * (5 - digits)
            return f"{lo:05d}" if lo == hi else f"{lo:05d}-{hi:05d}"
        return str(lo) if lo == hi else f"{lo}-{hi}"

    def parse(self, text: str):
        text = text.strip()
        if self.kind == "categorical":
            return frozenset(text.split("|"))
        if self.kind == "date":
            a, _, b = text.partition("..")
            lo = dt.date.fromisoformat(a).toordinal()
            return (lo, dt.date.fromisoformat(b).toordinal() if b else lo)
        if self.kind == "zip":
            if "*" in text:
                prefix = text.rstrip("*")
                stars = len(text) - len(prefix)
                lo = int(prefix or "0") * 10 ** stars
                return (lo, lo + 10 ** stars - 1)
            a, _, b = text.partition("-")
            return (int(a), int(b) if b else int(a))
        # numeric: allow negative lower bound "-5-3"
        head, sep, tail = text[1:].partition("-")
        if sep:
            return (int(text[0] + head), int(tail))
        return (int(text), int(text))
