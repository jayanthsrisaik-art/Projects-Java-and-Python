"""Apply a delta to a basis and verify the result end to end."""
from __future__ import annotations

from typing import BinaryIO

from .delta import Copy, Delta
from .errors import DeltaError
from .signature import file_hash


def _check(delta: Delta, basis_len: int) -> None:
    total = 0
    for i, op in enumerate(delta.ops):
        if isinstance(op, Copy) and (op.offset < 0 or op.offset + op.length > basis_len):
            raise DeltaError(
                f"op {i}: COPY [{op.offset}, {op.offset + op.length}) outside basis of "
                f"{basis_len} bytes (wrong basis file?)")
        total += op.length
    if total != delta.target_len:
        raise DeltaError(f"ops produce {total} bytes but header says {delta.target_len}")


def apply_delta(basis: bytes, delta: Delta) -> bytes:
    _check(delta, len(basis))
    out = bytearray()
    for op in delta.ops:
        if isinstance(op, Copy):
            out += basis[op.offset:op.offset + op.length]
        else:
            out += op.data
    result = bytes(out)
    if file_hash(result) != delta.target_hash:
        raise DeltaError("reconstructed file hash mismatch (basis changed or block collision)")
    return result


def apply_delta_stream(basis: BinaryIO, basis_len: int, delta: Delta, out: BinaryIO) -> None:
    """Streaming variant: seeks in ``basis`` and writes to ``out`` without loading the basis."""
    from hashlib import blake2b

    _check(delta, basis_len)
    h = blake2b(digest_size=32)
    for op in delta.ops:
        if isinstance(op, Copy):
            basis.seek(op.offset)
            data = basis.read(op.length)
            if len(data) != op.length:
                raise DeltaError("basis shorter than expected while streaming")
        else:
            data = op.data
        h.update(data)
        out.write(data)
    if h.digest() != delta.target_hash:
        raise DeltaError("reconstructed file hash mismatch (basis changed or block collision)")
