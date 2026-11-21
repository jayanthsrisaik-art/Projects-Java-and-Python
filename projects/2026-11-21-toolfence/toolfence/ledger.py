"""The context ledger: every piece of text the model has seen, with its source
and trust level, in arrival order.

The guard never sees the model's internal reasoning, so it cannot know how a
value got into an argument. What it *can* know is where that value could have
come from: if a recipient address appears in the user's message or in a trusted
tool result, the model could have taken it from there; if it appears only in a
web page or an email body, the model can only have taken it from content an
attacker may control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from .atoms import extract

TRUST_RANK = {"user": 3, "trusted": 2, "untrusted": 1}


@dataclass
class Segment:
    index: int
    source: str          # "user" or "tool:<name>"
    trust: str           # "user" | "trusted" | "untrusted"
    text: str
    secret: bool = False

    @cached_property
    def _atoms(self) -> dict[str, set[str]]:
        return {}

    def atoms(self, kind: str) -> set[str]:
        cache = self._atoms
        if kind not in cache:
            cache[kind] = extract(kind, self.text)
        return cache[kind]


@dataclass
class Origin:
    trust: str | None                 # best trust level where the atom appears, None if nowhere
    sources: list[str] = field(default_factory=list)  # sources at that trust level


@dataclass
class Ledger:
    segments: list[Segment] = field(default_factory=list)

    def add(self, source: str, trust: str, text: str, secret: bool = False) -> Segment:
        if trust not in TRUST_RANK:
            raise ValueError(f"unknown trust level: {trust}")
        seg = Segment(len(self.segments), source, trust, text, secret)
        self.segments.append(seg)
        return seg

    def origin(self, kind: str, atom: str) -> Origin:
        best: str | None = None
        sources: list[str] = []
        for seg in self.segments:
            if atom not in seg.atoms(kind):
                continue
            if best is None or TRUST_RANK[seg.trust] > TRUST_RANK[best]:
                best, sources = seg.trust, [seg.source]
            elif seg.trust == best and seg.source not in sources:
                sources.append(seg.source)
        return Origin(best, sources)

    def secrets(self) -> list[Segment]:
        return [s for s in self.segments if s.secret]
