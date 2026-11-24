"""Replicated Growable Array (RGA) text CRDT.

Each character is a node with a globally unique id ``(lamport, replica)``.
An insert names the node it was typed *after* (its ``origin``); concurrent
inserts after the same origin are ordered by id, larger id first.  Deletes
only set a tombstone, so every replica can still resolve later inserts that
reference a deleted character.

Operations travel between replicas as plain dicts (JSON-serialisable):

    {"kind": "ins", "rep": "a", "seq": 3, "id": [7, "a"], "origin": [6, "a"], "ch": "x"}
    {"kind": "del", "rep": "b", "seq": 1, "target": [7, "a"]}

``rep``/``seq`` identify the operation itself (``seq`` is a per-replica
counter starting at 1 with no gaps) and drive version vectors and causal
delivery.  ``id`` is the character's ordering key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

NodeId = Tuple[int, str]
ROOT: NodeId = (0, "")


class CRDTError(Exception):
    """Raised for invalid local edits or malformed operations."""


BLOCK_MAX = 128  # nodes per block before it is split in half


class _Node:
    __slots__ = ("id", "ch", "origin", "deleted", "blk")

    def __init__(self, nid: NodeId, ch: str, origin: NodeId):
        self.id, self.ch, self.origin = nid, ch, origin
        self.deleted = False
        self.blk: Optional["_Block"] = None


class _Block:
    """A run of consecutive nodes plus its count of visible characters.

    The document is a list of blocks.  Finding a visible position skips whole
    blocks by their counts, so lookups cost O(n/B + B) instead of O(n).
    """

    __slots__ = ("nodes", "visible")

    def __init__(self, nodes: List[_Node]):
        self.nodes = nodes
        self.visible = sum(1 for n in nodes if not n.deleted)
        for n in nodes:
            n.blk = self


def _as_id(value) -> NodeId:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lam, rep = value
        if isinstance(lam, int) and not isinstance(lam, bool) and isinstance(rep, str):
            return (lam, rep)
    raise CRDTError(f"malformed node id: {value!r}")


def validate_op(op: dict) -> dict:
    """Check the shape of a remote operation; return it unchanged if valid."""
    if not isinstance(op, dict):
        raise CRDTError(f"operation must be an object, got {type(op).__name__}")
    kind = op.get("kind")
    rep, seq = op.get("rep"), op.get("seq")
    if not isinstance(rep, str) or not rep:
        raise CRDTError(f"operation has invalid 'rep': {rep!r}")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
        raise CRDTError(f"operation has invalid 'seq': {seq!r}")
    if kind == "ins":
        nid = _as_id(op.get("id"))
        _as_id(op.get("origin"))
        ch = op.get("ch")
        if nid[1] != rep:
            raise CRDTError("insert id replica must equal the operation's 'rep'")
        if nid[0] < 1:
            raise CRDTError("insert lamport must be >= 1")
        if not isinstance(ch, str) or len(ch) != 1:
            raise CRDTError(f"insert 'ch' must be one character, got {ch!r}")
        if _as_id(op["origin"]) != ROOT and _as_id(op["origin"])[0] >= nid[0]:
            raise CRDTError("insert lamport must exceed its origin's lamport")
    elif kind == "del":
        _as_id(op.get("target"))
        if _as_id(op["target"]) == ROOT:
            raise CRDTError("cannot delete the root")
    else:
        raise CRDTError(f"unknown operation kind: {kind!r}")
    return op


@dataclass
class ReceiveReport:
    applied: int = 0
    duplicates: int = 0
    buffered: int = 0  # still waiting for causal dependencies after this call


class Replica:
    """One site's copy of the document."""

    def __init__(self, replica_id: str):
        if not isinstance(replica_id, str) or not replica_id:
            raise CRDTError("replica id must be a non-empty string")
        self.replica_id = replica_id
        self._blocks: List[_Block] = []
        self._nodes: Dict[NodeId, _Node] = {}  # every inserted char, incl. tombstones
        self._lamport = 0
        self._seq = 0
        self._visible = 0
        # Applied operations per origin replica, in seq order (index = seq-1).
        self._log: Dict[str, List[dict]] = {}
        self._pending: Dict[Tuple[str, int], dict] = {}
        self._waiting: Dict[tuple, List[dict]] = {}  # blocker -> parked ops
        self._order: List[dict] = []  # application order (always causal)

    # ------------------------------------------------------------------ views
    def text(self) -> str:
        return "".join(n.ch for n in self._iter_nodes() if not n.deleted)

    def __len__(self) -> int:
        return self._visible

    def version(self) -> Dict[str, int]:
        """Version vector: highest contiguous seq applied per replica."""
        return {rep: len(ops) for rep, ops in self._log.items() if ops}

    def pending_count(self) -> int:
        return len(self._pending)

    def stats(self) -> dict:
        total = len(self._nodes)
        return {
            "blocks": len(self._blocks),
            "visible_chars": self._visible,
            "tombstones": total - self._visible,
            "ops_applied": sum(len(v) for v in self._log.values()),
            "pending": len(self._pending),
            "lamport": self._lamport,
        }

    def _iter_nodes(self, bi: int = 0, i: int = 0) -> Iterator[_Node]:
        """Nodes in document order starting at block ``bi``, offset ``i``."""
        for blk in self._blocks[bi:]:
            yield from blk.nodes[i:]
            i = 0

    def _locate(self, node: _Node) -> Tuple[int, int]:
        blk = node.blk
        return self._blocks.index(blk), blk.nodes.index(node)

    def _visible_slot(self, pos: int) -> Tuple[int, int]:
        """(block, offset) of the visible node at 0-based visible index ``pos``."""
        if 0 <= pos < self._visible:
            for bi, blk in enumerate(self._blocks):
                if pos < blk.visible:
                    for i, node in enumerate(blk.nodes):
                        if not node.deleted:
                            if pos == 0:
                                return bi, i
                            pos -= 1
                pos -= blk.visible
        raise CRDTError(f"position {pos} out of range (length {self._visible})")

    def _visible_node_at(self, pos: int) -> _Node:
        bi, i = self._visible_slot(pos)
        return self._blocks[bi].nodes[i]

    # ------------------------------------------------------------ local edits
    def insert(self, pos: int, text: str) -> List[dict]:
        """Insert ``text`` before visible index ``pos``; return the new ops."""
        if not isinstance(pos, int) or pos < 0 or pos > self._visible:
            raise CRDTError(f"insert position {pos!r} out of range 0..{self._visible}")
        if not isinstance(text, str):
            raise CRDTError("inserted text must be a string")
        origin = ROOT if pos == 0 else self._visible_node_at(pos - 1).id
        ops = []
        for ch in text:
            self._lamport += 1
            self._seq += 1
            nid = (self._lamport, self.replica_id)
            op = {"kind": "ins", "rep": self.replica_id, "seq": self._seq,
                  "id": [nid[0], nid[1]], "origin": [origin[0], origin[1]], "ch": ch}
            self._apply(op)
            ops.append(op)
            origin = nid  # typing a run chains each char to the previous one
        return ops

    def delete(self, pos: int, length: int = 1) -> List[dict]:
        """Delete ``length`` visible characters starting at ``pos``."""
        if not isinstance(length, int) or length < 0:
            raise CRDTError("delete length must be a non-negative integer")
        if not isinstance(pos, int) or pos < 0 or pos + length > self._visible:
            raise CRDTError(f"delete range {pos}+{length} out of range (length {self._visible})")
        targets = []
        if length:
            for node in self._iter_nodes(*self._visible_slot(pos)):
                if not node.deleted:
                    targets.append(node.id)
                    if len(targets) == length:
                        break
        ops = []
        for target in targets:
            self._seq += 1
            op = {"kind": "del", "rep": self.replica_id, "seq": self._seq,
                  "target": [target[0], target[1]]}
            self._apply(op)
            ops.append(op)
        return ops

    # ----------------------------------------------------------- remote input
    def receive(self, ops: Iterable[dict]) -> ReceiveReport:
        """Integrate remote ops in any order, with duplicates tolerated.

        Ops whose causal dependencies are missing (an earlier op from the same
        replica, the insert's origin, or the delete's target) are buffered and
        applied as soon as the dependency arrives.
        """
        report = ReceiveReport()
        for op in ops:
            validate_op(op)
            key = (op["rep"], op["seq"])
            if op["seq"] <= len(self._log.get(op["rep"], ())) or key in self._pending:
                report.duplicates += 1
                continue
            self._pending[key] = op
            report.applied += self._try_apply(op)
        report.buffered = len(self._pending)
        return report

    def _blocker(self, op: dict):
        """The missing dependency of ``op``, or None if it can be applied."""
        if op["seq"] != len(self._log.get(op["rep"], ())) + 1:
            return ("seq", op["rep"], op["seq"] - 1)
        dep = _as_id(op["origin"] if op["kind"] == "ins" else op["target"])
        if dep != ROOT and dep not in self._nodes:
            return ("node", dep)
        return None

    def _try_apply(self, op: dict) -> int:
        """Apply ``op`` if ready, then anything it unblocks (worklist)."""
        applied = 0
        work = [op]
        while work:
            cur = work.pop()
            key = (cur["rep"], cur["seq"])
            if key not in self._pending:
                continue
            blocker = self._blocker(cur)
            if blocker is not None:
                self._waiting.setdefault(blocker, []).append(cur)
                continue
            del self._pending[key]
            self._apply(cur)
            applied += 1
            work.extend(self._waiting.pop(("seq", cur["rep"], cur["seq"]), ()))
            if cur["kind"] == "ins":
                work.extend(self._waiting.pop(("node", _as_id(cur["id"])), ()))
        return applied

    def _apply(self, op: dict) -> None:
        rep = op["rep"]
        log = self._log.setdefault(rep, [])
        if op["seq"] != len(log) + 1:  # pragma: no cover - guarded by callers
            raise CRDTError("operation applied out of causal order")
        if op["kind"] == "ins":
            nid, origin = _as_id(op["id"]), _as_id(op["origin"])
            if nid in self._nodes:
                raise CRDTError(f"node id {nid} already exists (replica id reused?)")
            self._integrate(_Node(nid, op["ch"], origin))
            self._lamport = max(self._lamport, nid[0])
        else:
            node = self._nodes[_as_id(op["target"])]
            if not node.deleted:
                node.deleted = True
                node.blk.visible -= 1
                self._visible -= 1
        log.append(op)
        self._order.append(op)
        if rep == self.replica_id:
            self._seq = max(self._seq, op["seq"])

    def _integrate(self, new: _Node) -> None:
        # RGA placement: start right after the origin and skip every node with
        # a larger id.  Those are concurrent siblings that win the tie (and
        # their descendants, whose lamport is larger still).  The first smaller
        # id marks the end of the origin's higher-priority subtrees.
        if not self._blocks:
            self._blocks.append(_Block([]))
        if new.origin == ROOT:
            bi, i = 0, 0
        else:
            bi, i = self._locate(self._nodes[new.origin])
            i += 1
        blocks = self._blocks
        while True:
            nodes = blocks[bi].nodes
            if i == len(nodes):
                if bi + 1 == len(blocks):
                    break  # end of document
                bi, i = bi + 1, 0
                continue
            if nodes[i].id > new.id:
                i += 1
            else:
                break
        blk = blocks[bi]
        blk.nodes.insert(i, new)
        new.blk = blk
        blk.visible += 1
        if len(blk.nodes) > BLOCK_MAX:
            half = len(blk.nodes) // 2
            tail = _Block(blk.nodes[half:])
            del blk.nodes[half:]
            blk.visible -= tail.visible
            blocks.insert(bi + 1, tail)
        self._nodes[new.id] = new
        self._visible += 1

    # ----------------------------------------------------------------- sync
    def ops_since(self, version: Optional[Dict[str, int]] = None) -> List[dict]:
        """All applied ops the holder of ``version`` has not seen yet.

        Returned in this replica's application order, which is causally valid,
        so a peer applies it in one pass (``receive`` also accepts any
        permutation and buffers what it cannot apply yet).
        """
        version = version or {}
        return [op for op in self._order if op["seq"] > version.get(op["rep"], 0)]

    # ---------------------------------------------------------- anchors
    def anchor(self, pos: int) -> List:
        """Stable reference to the gap before visible index ``pos``.

        Encoded as the id of the character to the left (ROOT for pos 0), so a
        cursor survives concurrent edits elsewhere in the document.
        """
        if not isinstance(pos, int) or pos < 0 or pos > self._visible:
            raise CRDTError(f"anchor position {pos!r} out of range 0..{self._visible}")
        nid = ROOT if pos == 0 else self._visible_node_at(pos - 1).id
        return [nid[0], nid[1]]

    def resolve(self, anchor) -> int:
        """Current visible index of an anchor created by :meth:`anchor`."""
        nid = _as_id(anchor)
        if nid == ROOT:
            return 0
        if nid not in self._nodes:
            raise CRDTError(f"unknown anchor {anchor!r}")
        bi, i = self._locate(self._nodes[nid])
        # Visible chars up to and including the anchored one.  If that char was
        # deleted, the gap sits after the nearest visible char to its left,
        # which is exactly this count.
        pos = sum(b.visible for b in self._blocks[:bi])
        return pos + sum(1 for n in self._blocks[bi].nodes[:i + 1] if not n.deleted)

    # ---------------------------------------------------------- persistence
    def snapshot_ops(self) -> List[dict]:
        return self.ops_since({})

    @classmethod
    def from_ops(cls, replica_id: str, ops: Iterable[dict]) -> "Replica":
        """Rebuild a replica (e.g. after restart) from a saved op log."""
        r = cls(replica_id)
        report = r.receive(ops)
        if report.buffered:
            raise CRDTError(f"op log is missing dependencies for {report.buffered} ops")
        return r
