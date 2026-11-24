"""Independent reference model of RGA used to cross-check the replica.

The linked-list replica places each insert with a single forward scan.  The
oracle instead builds the explicit causal tree (each char is a child of its
origin), sorts siblings by id descending and reads the text off a pre-order
walk.  The two are written differently on purpose; the test-suite and the
fuzzer assert they always agree.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from .doc import ROOT, NodeId


def oracle_text(ops: Iterable[dict]) -> str:
    children: Dict[NodeId, List[NodeId]] = {}
    chars: Dict[NodeId, str] = {}
    deleted = set()
    for op in ops:
        if op["kind"] == "ins":
            nid = (op["id"][0], op["id"][1])
            chars[nid] = op["ch"]
            children.setdefault((op["origin"][0], op["origin"][1]), []).append(nid)
        else:
            deleted.add((op["target"][0], op["target"][1]))
    out: List[str] = []
    stack = [ROOT]
    while stack:
        nid = stack.pop()
        if nid != ROOT and nid not in deleted:
            out.append(chars[nid])
        kids = sorted(children.get(nid, ()), reverse=True)
        stack.extend(reversed(kids))  # largest id visited first
    return "".join(out)
