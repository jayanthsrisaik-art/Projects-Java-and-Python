"""Turn one meter's raw register readings into validated, unwrapped segments.

A *segment* is a run of readings from one physical register with no
discontinuity. Inside a segment every reading has an unwrapped cumulative
value ``u`` (rollovers added back), so the energy between any two readings is
an exact difference. Readings are classified as:

- ``first``     first accepted reading of the meter
- ``ok``        consistent with the previous accepted reading
- ``rollover``  consistent only after adding REGISTER_MAX (register wrapped)
- ``misread``   inconsistent, while a later reading agrees with the previous one
                (and not with it - see ``_returns``)
- ``reset``     inconsistent and nothing later returns: a new segment starts
                (meter exchange or register reset; energy across it is unknown)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import GAP_S, LOOKAHEAD, MAX_KW, REGISTER_MAX, ROLLOVER_HIGH, ROLLOVER_LOW, TOL_KWH


@dataclass
class Interval:
    t0: int
    t1: int
    kwh: float

    @property
    def is_gap(self) -> bool:
        return self.t1 - self.t0 > GAP_S


@dataclass
class CleanResult:
    flags: list[tuple[int, float, str]]  # (ts, raw value, status) for every unique reading
    segments: list[list[tuple[int, float]]]  # accepted (ts, unwrapped kWh) per segment
    intervals: list[Interval] = field(default_factory=list)

    def issues(self) -> list[dict]:
        out = []
        for ts, raw, status in self.flags:
            if status in ("misread", "rollover", "reset"):
                out.append({"kind": status, "start_ts": ts, "end_ts": ts, "value": raw})
        for iv in self.intervals:
            if iv.is_gap:
                out.append({"kind": "gap", "start_ts": iv.t0, "end_ts": iv.t1, "value": round(iv.kwh, 3)})
        for a, b in zip(self.segments, self.segments[1:]):
            out.append({"kind": "unmetered", "start_ts": a[-1][0], "end_ts": b[0][0], "value": None})
        out.sort(key=lambda d: (d["start_ts"], d["kind"]))
        return out


def step(t_a: int, raw_a: float, t_b: int, raw_b: float) -> tuple[bool, float, bool]:
    """Is ``raw_b`` at ``t_b`` a plausible successor of ``raw_a`` at ``t_a``?

    Returns (plausible, energy delta in kWh, wrapped).
    """
    d = raw_b - raw_a
    wrapped = False
    if d < -TOL_KWH and raw_a > ROLLOVER_HIGH and raw_b < ROLLOVER_LOW:
        d += REGISTER_MAX
        wrapped = True
    hours = (t_b - t_a) / 3600.0
    ok = -TOL_KWH <= d <= MAX_KW * hours + TOL_KWH
    return ok, max(d, 0.0), wrapped


def _returns(lt: int, lv: float, t: int, v: float, ta: int, va: float) -> bool:
    """Does a later reading ``va`` show that ``v`` was a lone outlier?

    It must agree with the last accepted reading. If it agrees only through a
    register wrap *and* also agrees with ``v``, then ``v`` starts a coherent new
    regime (e.g. a meter exchanged while its register read 99,9xx) - not a misread.
    """
    ok, _, wrapped = step(lt, lv, ta, va)
    if not ok:
        return False
    return not wrapped or not step(t, v, ta, va)[0]


def clean_meter(rows: list[tuple[int, float]]) -> CleanResult:
    """Classify readings. ``rows`` must be unique by timestamp; order does not matter."""
    rows = sorted(rows)
    flags: list[tuple[int, float, str]] = []
    segments: list[list[tuple[int, float]]] = []
    last: tuple[int, float, float] | None = None  # (ts, raw, unwrapped)
    n = len(rows)
    for i, (t, v) in enumerate(rows):
        if last is None:
            segments.append([(t, 0.0)])
            last = (t, v, 0.0)
            flags.append((t, v, "first"))
            continue
        lt, lv, lu = last
        ok, d, wrapped = step(lt, lv, t, v)
        ahead = rows[i + 1 : i + 1 + LOOKAHEAD]
        if ok:
            # A plausible-looking value can still be a misread (e.g. +1000 kWh over a
            # long outage): it is one if the next reading contradicts it but agrees
            # with the last accepted reading.
            # (Agreement that needs a register wrap does not count: too ambiguous.)
            if ahead and not step(t, v, *ahead[0])[0] and step(lt, lv, *ahead[0])[:3:2] == (True, False):
                flags.append((t, v, "misread"))
                continue
            u = lu + d
            segments[-1].append((t, u))
            last = (t, v, u)
            flags.append((t, v, "rollover" if wrapped else "ok"))
        elif any(_returns(lt, lv, t, v, ta, va) for ta, va in ahead):
            flags.append((t, v, "misread"))
        else:
            segments.append([(t, 0.0)])
            last = (t, v, 0.0)
            flags.append((t, v, "reset"))

    intervals = []
    for seg in segments:
        for (t0, u0), (t1, u1) in zip(seg, seg[1:]):
            intervals.append(Interval(t0, t1, u1 - u0))
    return CleanResult(flags=flags, segments=segments, intervals=intervals)


def naive_intervals(rows: list[tuple[int, float]]) -> list[Interval]:
    """Baseline: difference consecutive readings in time order, clip negatives to zero."""
    rows = sorted(rows)
    return [Interval(t0, t1, max(v1 - v0, 0.0)) for (t0, v0), (t1, v1) in zip(rows, rows[1:])]
