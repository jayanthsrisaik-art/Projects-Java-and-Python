"""Basis-side block signatures (what the receiver sends to the sender).

The basis file is cut into fixed blocks of ``block_size`` bytes (the last block may be
shorter). Each block gets a 32-bit weak rolling checksum and a truncated BLAKE2b strong
hash. Serialized layout (all integers are LEB128 varints unless noted):

    b"DSSG" | version=1 | block_size | strong_len | file_len |
    per block: weak (4 bytes little-endian) + strong (strong_len bytes)

The block count is implied by file_len and block_size.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from hashlib import blake2b

from . import varint
from .errors import FormatError
from .rollsum import weak_checksum

MAGIC = b"DSSG"
VERSION = 1
DEFAULT_STRONG_LEN = 8
MIN_BLOCK = 700          # rsync's BLOCK_SIZE floor
MAX_BLOCK = 128 * 1024   # rsync's MAX_BLOCK_SIZE (protocol >= 30)


def default_block_size(file_len: int) -> int:
    """rsync-style heuristic: ~sqrt(file_len), multiple of 8, clamped to [700, 128 KiB]."""
    if file_len <= MIN_BLOCK * MIN_BLOCK:
        return MIN_BLOCK
    size = int(math.isqrt(file_len)) & ~7
    return max(MIN_BLOCK, min(MAX_BLOCK, size))


def strong_hash(data: bytes, strong_len: int) -> bytes:
    return blake2b(data, digest_size=strong_len).digest()


def file_hash(data: bytes) -> bytes:
    return blake2b(data, digest_size=32).digest()


@dataclass
class Signature:
    block_size: int
    strong_len: int
    file_len: int
    weak: list[int] = field(default_factory=list)
    strong: list[bytes] = field(default_factory=list)

    @property
    def block_count(self) -> int:
        return len(self.weak)

    def block_len(self, index: int) -> int:
        start = index * self.block_size
        return min(self.block_size, self.file_len - start)

    def to_bytes(self) -> bytes:
        out = bytearray(MAGIC)
        out += varint.encode(VERSION)
        out += varint.encode(self.block_size)
        out += varint.encode(self.strong_len)
        out += varint.encode(self.file_len)
        for w, s in zip(self.weak, self.strong):
            out += w.to_bytes(4, "little")
            out += s
        return bytes(out)

    @classmethod
    def from_bytes(cls, buf: bytes) -> "Signature":
        r = varint.Reader(buf)
        if r.take(4) != MAGIC:
            raise FormatError("not a deltasync signature (bad magic)")
        version = r.varint()
        if version != VERSION:
            raise FormatError(f"unsupported signature version {version}")
        block_size = r.varint()
        strong_len = r.varint()
        file_len = r.varint()
        _validate_params(block_size, strong_len)
        count = math.ceil(file_len / block_size) if file_len else 0
        sig = cls(block_size, strong_len, file_len)
        for _ in range(count):
            sig.weak.append(int.from_bytes(r.take(4), "little"))
            sig.strong.append(r.take(strong_len))
        r.expect_end()
        return sig


def _validate_params(block_size: int, strong_len: int) -> None:
    if block_size < 1:
        raise FormatError("block_size must be >= 1")
    if not 1 <= strong_len <= 32:
        raise FormatError("strong_len must be in [1, 32]")


def make_signature(basis: bytes, block_size: int | None = None,
                   strong_len: int = DEFAULT_STRONG_LEN) -> Signature:
    if block_size is None:
        block_size = default_block_size(len(basis))
    _validate_params(block_size, strong_len)
    sig = Signature(block_size, strong_len, len(basis))
    for start in range(0, len(basis), block_size):
        block = basis[start:start + block_size]
        sig.weak.append(weak_checksum(block))
        sig.strong.append(strong_hash(block, strong_len))
    return sig
