"""Answering strategies.

* ``naive``     – plain RAG: top BM25 chunks, read the value from the best one. No notion of time or status.
* ``latest``    – the common "fix": pick the policy family, then always use the newest version that is
                  active and already effective today.  Right for "now" questions, wrong for history.
* ``asof``      – this project: temporal intent parsing + validity intervals over *active* versions,
                  ranges that report every value in force, explicit abstention.
* ``asof_nostatus`` – ablation of ``asof`` that ignores the status field (drafts/withdrawn count).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

from .index import Corpus, Doc
from .reader import ExtractiveReader, Reader
from .temporal import Intent, parse, strip_temporal


@dataclass
class Evidence:
    doc_id: str
    value: str | None
    valid_from: str
    valid_to: str | None  # None = open-ended


@dataclass
class Answer:
    kind: str  # values | date
    values: list[str] = field(default_factory=list)
    date: str | None = None
    abstain: bool = False
    policy: str | None = None
    evidence: list[Evidence] = field(default_factory=list)
    intent: str = ""
    text: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "values": self.values, "date": self.date, "abstain": self.abstain,
            "policy": self.policy, "intent": self.intent, "text": self.text,
            "evidence": [e.__dict__ for e in self.evidence],
        }


def pick_family(corpus: Corpus, query: str, k: int = 8) -> str | None:
    votes: dict[str, float] = defaultdict(float)
    for ch, s in corpus.bm25.search(query, k):
        votes[ch.doc.policy] += s
    if not votes:
        return None
    return max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]


def intervals(docs: list[Doc]) -> list[tuple[Doc, date, date | None]]:
    """Validity interval [effective, next.effective - 1 day] for a chronologically sorted list."""
    out = []
    for i, d in enumerate(docs):
        nxt = docs[i + 1].effective - timedelta(days=1) if i + 1 < len(docs) else None
        out.append((d, d.effective, nxt))
    return out


def _collapse(vals: list[str | None]) -> list[str]:
    out: list[str] = []
    for v in vals:
        if v is not None and (not out or out[-1] != v):
            out.append(v)
    return out


class Answerer:
    def __init__(self, corpus: Corpus, ref: date, strategy: str = "asof", reader: Reader | None = None):
        if strategy not in ("naive", "latest", "asof", "asof_nostatus"):
            raise ValueError(strategy)
        self.corpus, self.ref, self.strategy = corpus, ref, strategy
        self.reader = reader or ExtractiveReader()

    # ----------------------------------------------------------------- helpers
    def _timeline(self, policy: str) -> list[Doc]:
        fam = self.corpus.family[policy]
        if self.strategy == "asof_nostatus":
            return list(fam)
        return [d for d in fam if d.status == "active"]

    def _read(self, q: str, d: Doc) -> str | None:
        return self.reader.read_value(q, d, self.corpus)

    def _render(self, a: Answer) -> Answer:
        if a.abstain:
            a.text = ("No version of this policy was in force for that time." if a.kind == "values"
                      else "No version of this policy ever set that value.")
            if a.policy is None:
                a.text = "I could not find a matching policy."
            return a
        cite = lambda e: f"[{e.doc_id}, from {e.valid_from}" + (f" to {e.valid_to}]" if e.valid_to else "]")  # noqa: E731
        if a.kind == "date":
            a.text = f"It took effect on {a.date} {cite(a.evidence[0])}."
        elif len(a.evidence) == 1:
            a.text = f"{a.values[0]} {cite(a.evidence[0])}."
        else:
            parts = [f"{e.value} {cite(e)}" for e in a.evidence]
            a.text = "The value changed during that period: " + "; then ".join(parts) + "."
        return a

    # ----------------------------------------------------------------- strategies
    def answer(self, question: str) -> Answer:
        intent = parse(question, self.ref)
        kind = "date" if intent.kind == "when" else "values"
        if self.strategy == "naive":
            return self._render(self._naive(question, intent, kind))
        query = strip_temporal(question, intent)
        policy = pick_family(self.corpus, query)
        if policy is None:
            return self._render(Answer(kind, abstain=True, intent=intent.kind))
        if self.strategy == "latest":
            return self._render(self._latest(question, intent, kind, policy))
        return self._render(self._asof(question, intent, kind, policy))

    def _naive(self, q: str, intent: Intent, kind: str) -> Answer:
        hits = self.corpus.bm25.search(q, 3)
        if not hits:
            return Answer(kind, abstain=True, intent=intent.kind)
        doc = hits[0][0].doc
        ev = Evidence(doc.doc_id, None, doc.effective.isoformat(), None)
        if kind == "date":
            return Answer(kind, date=doc.effective.isoformat(), policy=doc.policy, evidence=[ev], intent=intent.kind)
        ev.value = self._read(q, doc)
        return Answer(kind, values=_collapse([ev.value]), abstain=ev.value is None, policy=doc.policy,
                      evidence=[ev], intent=intent.kind)

    def _latest(self, q: str, intent: Intent, kind: str, policy: str) -> Answer:
        live = [d for d in self.corpus.family[policy] if d.status == "active" and d.effective <= self.ref]
        if not live:
            return Answer(kind, abstain=True, policy=policy, intent=intent.kind)
        if kind == "date":
            for d in live:  # first active, already-effective version stating that value
                if self._read(q, d) == intent.value:
                    return Answer(kind, date=d.effective.isoformat(), policy=policy, intent=intent.kind,
                                  evidence=[Evidence(d.doc_id, intent.value, d.effective.isoformat(), None)])
            return Answer(kind, abstain=True, policy=policy, intent=intent.kind)
        d = live[-1]
        v = self._read(q, d)
        return Answer(kind, values=_collapse([v]), abstain=v is None, policy=policy, intent=intent.kind,
                      evidence=[Evidence(d.doc_id, v, d.effective.isoformat(), None)])

    def _asof(self, q: str, intent: Intent, kind: str, policy: str) -> Answer:
        tl = intervals(self._timeline(policy))
        mk = lambda d, s, e, v: Evidence(d.doc_id, v, s.isoformat(), e.isoformat() if e else None)  # noqa: E731
        if intent.kind == "when":
            for d, s, e in tl:
                if self._read(q, d) == intent.value:
                    return Answer(kind, date=s.isoformat(), policy=policy, intent=intent.kind,
                                  evidence=[mk(d, s, e, intent.value)])
            return Answer(kind, abstain=True, policy=policy, intent=intent.kind)
        if intent.kind == "previous":
            cur = [i for i, (d, s, e) in enumerate(tl) if s <= self.ref]
            if len(cur) < 2:
                return Answer(kind, abstain=True, policy=policy, intent=intent.kind)
            sel = [tl[cur[-1] - 1]]
        else:
            sel = [(d, s, e) for d, s, e in tl if s <= intent.end and (e is None or e >= intent.start)]
        if not sel:
            return Answer(kind, abstain=True, policy=policy, intent=intent.kind)
        ev = [mk(d, s, e, self._read(q, d)) for d, s, e in sel]
        vals = _collapse([e.value for e in ev])
        return Answer(kind, values=vals, abstain=not vals, policy=policy, evidence=ev, intent=intent.kind)
