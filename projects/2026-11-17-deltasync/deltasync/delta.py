"""Target-side delta generation: find basis blocks anywhere in the target.

The sender slides a window over the target one byte at a time, updating the weak
checksum in O(1). A weak hit is confirmed with the strong hash before it is trusted.
Matches become COPY ops (basis byte ranges); everything between matches is LITERAL.

Wire format (varints unless noted):

    b"DSDL" | version=1 | target_len | target_hash (32 bytes BLAKE2b) | ops... | 0x00
    op COPY    = 0x01 | basis_offset | length
    op LITERAL = 0x02 | length | raw bytes
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import varint
from .errors import FormatError
from .rollsum import MASK, weak_parts
from .signature import Signature, file_hash, strong_hash

MAGIC = b"DSDL"
VERSION = 1
OP_END, OP_COPY, OP_LITERAL = 0x00, 0x01, 0x02


@dataclass(frozen=True)
class Copy:
    offset: int
    length: int


@dataclass(frozen=True)
class Literal:
    data: bytes

    @property
    def length(self) -> int:
        return len(self.data)


@dataclass
class DeltaStats:
    matched_blocks: int = 0
    weak_hits: int = 0
    false_weak_hits: int = 0   # weak checksum matched, strong hash did not

    def as_dict(self) -> dict:
        return {"matched_blocks": self.matched_blocks, "weak_hits": self.weak_hits,
                "false_weak_hits": self.false_weak_hits}


@dataclass
class Delta:
    target_len: int
    target_hash: bytes
    ops: list = field(default_factory=list)
    stats: DeltaStats = field(default_factory=DeltaStats)

    @property
    def copy_bytes(self) -> int:
        return sum(op.length for op in self.ops if isinstance(op, Copy))

    @property
    def literal_bytes(self) -> int:
        return sum(op.length for op in self.ops if isinstance(op, Literal))

    def to_bytes(self) -> bytes:
        out = bytearray(MAGIC)
        out += varint.encode(VERSION)
        out += varint.encode(self.target_len)
        out += self.target_hash
        for op in self.ops:
            if isinstance(op, Copy):
                out.append(OP_COPY)
                out += varint.encode(op.offset)
                out += varint.encode(op.length)
            else:
                out.append(OP_LITERAL)
                out += varint.encode(len(op.data))
                out += op.data
        out.append(OP_END)
        return bytes(out)

    @classmethod
    def from_bytes(cls, buf: bytes) -> "Delta":
        r = varint.Reader(buf)
        if r.take(4) != MAGIC:
            raise FormatError("not a deltasync delta (bad magic)")
        version = r.varint()
        if version != VERSION:
            raise FormatError(f"unsupported delta version {version}")
        delta = cls(r.varint(), r.take(32))
        while True:
            tag = r.byte()
            if tag == OP_END:
                break
            if tag == OP_COPY:
                offset = r.varint()
                length = r.varint()
                if length == 0:
                    raise FormatError("zero-length COPY")
                delta.ops.append(Copy(offset, length))
            elif tag == OP_LITERAL:
                length = r.varint()
                if length == 0:
                    raise FormatError("zero-length LITERAL")
                delta.ops.append(Literal(r.take(length)))
            else:
                raise FormatError(f"unknown op tag 0x{tag:02x} at offset {r.pos - 1}")
        r.expect_end()
        return delta


class _OpBuilder:
    """Appends ops, merging adjacent COPYs and adjacent LITERALs."""

    def __init__(self) -> None:
        self.ops: list = []

    def copy(self, offset: int, length: int) -> None:
        last = self.ops[-1] if self.ops else None
        if isinstance(last, Copy) and last.offset + last.length == offset:
            self.ops[-1] = Copy(last.offset, last.length + length)
        else:
            self.ops.append(Copy(offset, length))

    def literal(self, data: bytes) -> None:
        if not data:
            return
        last = self.ops[-1] if self.ops else None
        if isinstance(last, Literal):
            self.ops[-1] = Literal(last.data + data)
        else:
            self.ops.append(Literal(data))


def generate_delta(sig: Signature, target: bytes) -> Delta:
    n = len(target)
    bs = sig.block_size
    out = _OpBuilder()
    stats = DeltaStats()
    delta = Delta(n, file_hash(target), out.ops, stats)
    if n == 0:
        return delta
    if sig.block_count == 0:
        out.literal(target)
        return delta

    # Full-size blocks are looked up while the window is full; a short tail block can
    # only match the final bytes of the target (the window shrinks at the end).
    table: dict[int, list[int]] = {}
    tail_index = -1
    for i, w in enumerate(sig.weak):
        if sig.block_len(i) == bs:
            table.setdefault(w, []).append(i)
        else:
            tail_index = i
    tail_len = sig.block_len(tail_index) if tail_index >= 0 else -1

    strong_len = sig.strong_len
    strongs = sig.strong
    lookup = table.get

    k = 0
    lit_start = 0
    wlen = min(bs, n)
    a, b = weak_parts(target[0:wlen])
    next_block = -1  # block index that would continue the previous match

    while k < n:
        matched = -1
        if wlen == bs:
            cands = lookup(a | (b << 16))
            if cands is not None:
                stats.weak_hits += 1
                s = strong_hash(target[k:k + bs], strong_len)
                if next_block in cands and strongs[next_block] == s:
                    matched = next_block
                else:
                    for idx in cands:
                        if strongs[idx] == s:
                            matched = idx
                            break
                if matched < 0:
                    stats.false_weak_hits += 1
        elif wlen == tail_len and sig.weak[tail_index] == (a | (b << 16)):
            stats.weak_hits += 1
            if strongs[tail_index] == strong_hash(target[k:k + wlen], strong_len):
                matched = tail_index
            else:
                stats.false_weak_hits += 1

        if matched >= 0:
            out.literal(target[lit_start:k])
            out.copy(matched * bs, wlen)
            stats.matched_blocks += 1
            next_block = matched + 1
            k += wlen
            lit_start = k
            if k < n:
                wlen = min(bs, n - k)
                a, b = weak_parts(target[k:k + wlen])
            continue

        x = target[k]
        if k + wlen < n:
            y = target[k + wlen]
            a = (a - x + y) & MASK
            b = (b - wlen * x + a) & MASK
        else:
            a = (a - x) & MASK
            b = (b - wlen * x) & MASK
            wlen -= 1
        k += 1

    out.literal(target[lit_start:n])
    return delta
