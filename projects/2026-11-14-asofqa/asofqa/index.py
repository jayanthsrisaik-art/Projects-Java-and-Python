"""Document loading, paragraph chunking and a small BM25 index (stdlib only)."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .textutil import tokens

_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)


@dataclass(frozen=True)
class Doc:
    doc_id: str
    policy: str
    title: str
    version: int
    status: str
    effective: date
    body: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc: Doc
    text: str


def parse_doc(text: str) -> Doc:
    m = _FRONT.match(text)
    if not m:
        raise ValueError("document has no front matter")
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    required = ("doc_id", "policy", "title", "version", "status", "effective")
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"front matter missing {missing}")
    if meta["status"] not in ("active", "proposed", "withdrawn"):
        raise ValueError(f"unknown status {meta['status']!r}")
    return Doc(
        doc_id=meta["doc_id"], policy=meta["policy"], title=meta["title"], version=int(meta["version"]),
        status=meta["status"], effective=date.fromisoformat(meta["effective"]), body=text[m.end():].strip(),
    )


def load_docs(doc_dir: Path) -> list[Doc]:
    return [parse_doc(p.read_text(encoding="utf-8")) for p in sorted(Path(doc_dir).glob("*.md"))]


def chunk_doc(doc: Doc) -> list[Chunk]:
    paras = [p.strip() for p in doc.body.split("\n\n") if p.strip()]
    out = []
    for i, p in enumerate(paras):
        p = re.sub(r"^#[^\n]*(?:\n+|$)", "", p).strip()  # drop the heading line; the title is prefixed below
        if p:
            # "Contextual header": every chunk carries its document title so it is retrievable alone.
            out.append(Chunk(f"{doc.doc_id}#{i}", doc, f"{doc.title}. {p}"))
    return out


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(c.text)) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df: dict[str, int] = defaultdict(int)
        for t in self.tf:
            for term in t:
                df[term] += 1
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        q = tokens(query)
        scored = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for term in q:
                f = tf.get(term)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [(self.chunks[i], s) for i, s in scored[:k]]


class Corpus:
    """All documents plus the chunk index, grouped by policy family."""

    def __init__(self, docs: list[Doc]):
        self.docs = {d.doc_id: d for d in docs}
        self.chunks = [c for d in docs for c in chunk_doc(d)]
        self.chunks_by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for c in self.chunks:
            self.chunks_by_doc[c.doc.doc_id].append(c)
        self.family: dict[str, list[Doc]] = defaultdict(list)
        for d in docs:
            self.family[d.policy].append(d)
        for fam in self.family.values():
            fam.sort(key=lambda d: (d.effective, d.version))
        self.bm25 = BM25(self.chunks)

    @classmethod
    def from_dir(cls, doc_dir: Path) -> "Corpus":
        return cls(load_docs(doc_dir))
