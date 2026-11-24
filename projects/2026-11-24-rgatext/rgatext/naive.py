"""Baseline for comparison: replay remote edits at their original indices.

This is what a hand-rolled "send the edit, apply it on the other side" sync
does.  It needs no metadata, but once two replicas edit concurrently the
indices refer to different documents and the copies diverge silently.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .doc import CRDTError, ReceiveReport


class NaiveReplica:
    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self._text = ""
        self._seq = 0
        self._log: Dict[str, List[dict]] = {}
        self._order: List[dict] = []

    def text(self) -> str:
        return self._text

    def __len__(self) -> int:
        return len(self._text)

    def version(self) -> Dict[str, int]:
        """Highest contiguous seq seen per replica (arrival may have gaps)."""
        out = {}
        for rep, ops in self._log.items():
            seqs, n = {o["seq"] for o in ops}, 0
            while n + 1 in seqs:
                n += 1
            if n:
                out[rep] = n
        return out

    def pending_count(self) -> int:
        return 0

    def insert(self, pos: int, text: str) -> List[dict]:
        if pos < 0 or pos > len(self._text):
            raise CRDTError(f"insert position {pos} out of range 0..{len(self._text)}")
        return [self._local({"kind": "ins", "pos": pos, "text": text})]

    def delete(self, pos: int, length: int = 1) -> List[dict]:
        if pos < 0 or length < 0 or pos + length > len(self._text):
            raise CRDTError(f"delete range {pos}+{length} out of range (length {len(self._text)})")
        return [self._local({"kind": "del", "pos": pos, "len": length})]

    def _local(self, op: dict) -> dict:
        self._seq += 1
        op.update(rep=self.replica_id, seq=self._seq)
        self._apply(op)
        return op

    def _apply(self, op: dict) -> None:
        # Clamp indices so replay never crashes; it just lands in the wrong place.
        pos = min(op["pos"], len(self._text))
        if op["kind"] == "ins":
            self._text = self._text[:pos] + op["text"] + self._text[pos:]
        else:
            self._text = self._text[:pos] + self._text[pos + op["len"]:]
        self._log.setdefault(op["rep"], []).append(op)
        self._order.append(op)

    def receive(self, ops: Iterable[dict]) -> ReceiveReport:
        """Apply in arrival order; skip exact duplicates by (rep, seq)."""
        report = ReceiveReport()
        seen = {(o["rep"], o["seq"]) for o in self._order}
        for op in ops:
            if (op["rep"], op["seq"]) in seen:
                report.duplicates += 1
                continue
            seen.add((op["rep"], op["seq"]))
            self._apply(op)
            report.applied += 1
        return report

    def ops_since(self, version: Optional[Dict[str, int]] = None) -> List[dict]:
        version = version or {}
        return [op for op in self._order if op["seq"] > version.get(op["rep"], 0)]
