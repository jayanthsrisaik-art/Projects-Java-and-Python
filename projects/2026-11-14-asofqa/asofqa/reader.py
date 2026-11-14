"""Readers turn one document into the value it *states* for the question.

``ExtractiveReader`` is the default, fully local reader.  ``ClaudeReader`` is an optional drop-in that
asks Claude for the same thing; it is never needed to run the project or the tests.
"""
from __future__ import annotations

import os
import re
from typing import Protocol

from .index import Corpus, Doc
from .textutil import QUANTITY, norm_number, sentences, tokens

# Sentences that talk about an *older* value or about status, not the value this version sets.
HISTORY = re.compile(r"\b(previous|previously|replaces|replaced|formerly|prior|until then|used to|"
                     r"withdrawn|draft|would take effect)\b", re.I)
# Constraint/procedure sentences ("at least 10 days in advance") carry numbers but rarely the entitlement.
CONSTRAINT = re.compile(r"\b(must|should|at least|at most|no more than|more than|within|above|never|"
                        r"required?|notice|after)\b", re.I)


def _overlap_key(tok: str) -> str:
    """Crude truncation stemmer for overlap scoring: 'reimbursed' and 'reimbursement' both -> 'reimbu'."""
    return tok[:6]


TEMPORAL_WORDS = set(tokens(
    "current currently now today present these days ago year years month last previous before change "
    "recent most revised update throughout during starting since date become became applied apply effect "
    "employees get looking back half second first q1 q2 q3 q4 will used"
))


class Reader(Protocol):
    def read_value(self, question: str, doc: Doc, corpus: Corpus) -> str | None: ...


class ExtractiveReader:
    """Pick the best value-bearing sentence of a document and return its first unit quantity.

    Sentence score = overlap with the question's content words + 0.5 * overlap with the document
    title (the main quantity of a policy usually restates its subject) - 1 for constraint wording
    ("at least", "within", "must" ...), which marks procedural numbers rather than the entitlement.  Sentences that describe a
    replaced value or a status note are excluded unless nothing else carries a quantity.
    """

    def __init__(self, history_filter: bool = True):
        self.history_filter = history_filter

    def best_sentence(self, question: str, doc: Doc, corpus: Corpus) -> str | None:
        qterms = {_overlap_key(t) for t in set(tokens(question)) - TEMPORAL_WORDS}
        tterms = {_overlap_key(t) for t in set(tokens(doc.title)) - {"policy"}}
        text = " ".join(c.text.split(". ", 1)[1] if ". " in c.text else c.text
                        for c in corpus.chunks_by_doc[doc.doc_id])
        cands = []
        for pos, s in enumerate(sentences(text)):
            if not QUANTITY.search(s):
                continue
            st = {_overlap_key(t) for t in tokens(s)}
            score = len(st & qterms) + 0.5 * len(st & tterms) - (1.0 if CONSTRAINT.search(s) else 0.0)
            hist = bool(HISTORY.search(s))
            cands.append((hist if self.history_filter else False, -score, pos, s))
        if not cands:
            return None
        cands.sort()
        return cands[0][3]

    def read_value(self, question: str, doc: Doc, corpus: Corpus) -> str | None:
        s = self.best_sentence(question, doc, corpus)
        if s is None:
            return None
        m = QUANTITY.search(s)
        return norm_number(m.group("money") or m.group("num"))


class ClaudeReader:
    """Optional LLM reader (Anthropic SDK). Pass ``client`` to inject a fake in tests."""

    PROMPT = (
        "You are reading one version of a company policy document.\n"
        "Question: {question}\n\n<document>\n{doc}\n</document>\n\n"
        "Reply with only the single number that THIS document sets as its own value for the quantity "
        "the question asks about (not a previous or replaced value). No units, no currency symbols, no words. "
        "If the document does not set such a value, reply NONE."
    )

    def __init__(self, client=None, model: str = "claude-opus-5-5"):
        if client is None:
            import anthropic  # optional dependency

            client = anthropic.Anthropic()
        self.client = client
        self.model = model

    def read_value(self, question: str, doc: Doc, corpus: Corpus) -> str | None:
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=1024,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": self.PROMPT.format(question=question, doc=doc.body)}],
        )
        if getattr(resp, "stop_reason", None) == "refusal":
            return None
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip()
        m = re.search(r"\d[\d,]*(?:\.\d+)?", text)
        if not m or text.upper().startswith("NONE"):
            return None
        return norm_number(m.group(0))


def make_reader(name: str) -> Reader:
    if name == "extractive":
        return ExtractiveReader()
    if name == "claude":
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise SystemExit("--reader claude needs Anthropic credentials; use the default extractive reader offline")
        return ClaudeReader()
    raise ValueError(name)
