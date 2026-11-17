"""Unsigned LEB128 varints with strict decoding (no truncation, no overlong encodings)."""
from __future__ import annotations

from .errors import FormatError

MAX_BYTES = 10  # enough for 64-bit values


def encode(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint must be non-negative")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def decode(buf: bytes, pos: int) -> tuple[int, int]:
    """Return (value, new_pos)."""
    result = 0
    shift = 0
    for i in range(MAX_BYTES):
        if pos >= len(buf):
            raise FormatError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if byte == 0 and i > 0:
                raise FormatError("overlong varint encoding")
            return result, pos
        shift += 7
    raise FormatError("varint too long")


class Reader:
    """Cursor over a byte buffer used by the signature/delta decoders."""

    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.pos = 0

    def varint(self) -> int:
        value, self.pos = decode(self.buf, self.pos)
        return value

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.buf):
            raise FormatError(f"truncated: wanted {n} bytes at offset {self.pos}")
        chunk = self.buf[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def byte(self) -> int:
        return self.take(1)[0]

    def expect_end(self) -> None:
        if self.pos != len(self.buf):
            raise FormatError(f"{len(self.buf) - self.pos} trailing bytes")
