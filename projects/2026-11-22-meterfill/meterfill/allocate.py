"""Hourly allocation.

Energy between two consecutive accepted readings is *known* (it is a register
difference). The only question is how it is spread over the hours in between:

- short intervals (<= GAP_S): constant rate          -> quality ``actual``
- gaps with a closing reading: spread in proportion to an hourly shape
  prediction, then rescaled so the gap total is exactly the register delta
                                                      -> quality ``allocated``
- spans with no closing reading (before the first / after the last reading,
  across a meter exchange): the prediction itself     -> quality ``estimated``
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clean import CleanResult, Interval
from .config import ACTUAL, ALLOCATED, ESTIMATED, HOUR
from .timeutil import overlaps


@dataclass
class Coverage:
    h0: int
    nh: int
    actual_kwh: np.ndarray
    actual_sec: np.ndarray
    gaps: list[Interval] = field(default_factory=list)
    open_spans: list[tuple[int, int]] = field(default_factory=list)

    @property
    def full_actual(self) -> np.ndarray:
        """Hours entirely covered by short intervals: the only hours used to learn profiles."""
        return self.actual_sec >= HOUR - 1e-6

    def mean_rate(self) -> float:
        """Average kWh per hour over all metered time (short intervals and gaps)."""
        sec = self.actual_sec.sum() + sum(min(g.t1, self.h0 + self.nh * HOUR) - max(g.t0, self.h0) for g in self.gaps)
        kwh = self.actual_kwh.sum() + sum(g.kwh for g in self.gaps)
        return float(kwh / sec * HOUR) if sec > 0 else 0.5


def coverage(clean: CleanResult, h0: int, nh: int) -> Coverage:
    akwh = np.zeros(nh)
    asec = np.zeros(nh)
    gaps = []
    for iv in clean.intervals:
        if iv.is_gap:
            gaps.append(iv)
            continue
        idx, sec = overlaps(iv.t0, iv.t1, h0, nh)
        if idx.size:
            rate = iv.kwh / (iv.t1 - iv.t0)
            np.add.at(akwh, idx, rate * sec)
            np.add.at(asec, idx, sec)
    t_end = h0 + nh * HOUR
    spans = []
    if clean.segments:
        if clean.segments[0][0][0] > h0:
            spans.append((h0, clean.segments[0][0][0]))
        for a, b in zip(clean.segments, clean.segments[1:]):
            spans.append((a[-1][0], b[0][0]))
        if clean.segments[-1][-1][0] < t_end:
            spans.append((clean.segments[-1][-1][0], t_end))
    else:
        spans.append((h0, t_end))
    return Coverage(h0, nh, akwh, asec, gaps, [(a, b) for a, b in spans if b > a])


@dataclass
class Filled:
    kwh: np.ndarray
    quality: np.ndarray
    gap_id: np.ndarray  # index into Coverage.gaps for allocated hours, else -1


def fill(cov: Coverage, pred: np.ndarray, scale: bool = True) -> Filled:
    """Complete the hourly series. ``pred`` is a predicted kWh for every hour of the window.

    With ``scale=False`` gap hours get the raw prediction (ignores the known
    total) - kept only as an ablation that shows why conservation matters.
    """
    kwh = cov.actual_kwh.copy()
    quality = np.full(cov.nh, ACTUAL, dtype=np.int8)
    gap_id = np.full(cov.nh, -1, dtype=np.int64)
    pred = np.clip(np.asarray(pred, dtype=float), 1e-6, None)
    for gi, g in enumerate(cov.gaps):
        idx, sec = overlaps(g.t0, g.t1, cov.h0, cov.nh)
        if not idx.size:
            continue
        w = pred[idx] * sec / HOUR
        if scale:
            # Part of the gap may fall outside the window; scale by the share inside.
            share = sec.sum() / (g.t1 - g.t0)
            kwh[idx] += g.kwh * share * w / w.sum()
        else:
            kwh[idx] += w
        quality[idx] = np.maximum(quality[idx], ALLOCATED)
        gap_id[idx] = gi
    for a, b in cov.open_spans:
        idx, sec = overlaps(a, b, cov.h0, cov.nh)
        if idx.size:
            kwh[idx] += pred[idx] * sec / HOUR
            quality[idx] = ESTIMATED
    return Filled(kwh, quality, gap_id)


def fill_intervals_uniform(intervals: list[Interval], h0: int, nh: int) -> np.ndarray:
    """Spread each interval at constant rate (used by the naive baseline)."""
    out = np.zeros(nh)
    for iv in intervals:
        idx, sec = overlaps(iv.t0, iv.t1, h0, nh)
        if idx.size:
            np.add.at(out, idx, iv.kwh / (iv.t1 - iv.t0) * sec)
    return out
