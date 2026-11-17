"""rsync's weak rolling checksum.

For a window x_0..x_{L-1}:
    a = sum(x_i)               mod 2^16
    b = sum((L - i) * x_i)     mod 2^16
    digest = a | (b << 16)

Sliding the window one byte to the right (drop x_0, add x_L) is O(1):
    a' = a - x_0 + x_L
    b' = b - L * x_0 + a'
Shrinking the window from the left (used near the end of the file) is also O(1):
    a' = a - x_0
    b' = b - L * x_0
"""
from __future__ import annotations

MASK = 0xFFFF


def weak_parts(data: bytes) -> tuple[int, int]:
    n = len(data)
    a = 0
    b = 0
    for i, x in enumerate(data):
        a += x
        b += (n - i) * x
    return a & MASK, b & MASK


def weak_checksum(data: bytes) -> int:
    a, b = weak_parts(data)
    return a | (b << 16)


class RollingChecksum:
    """Explicit rolling-window object (the delta hot loop inlines the same arithmetic)."""

    __slots__ = ("a", "b", "length")

    def __init__(self, window: bytes = b"") -> None:
        self.a, self.b = weak_parts(window)
        self.length = len(window)

    @property
    def digest(self) -> int:
        return self.a | (self.b << 16)

    def roll(self, out_byte: int, in_byte: int) -> None:
        """Slide the window: drop ``out_byte`` on the left, append ``in_byte`` on the right."""
        self.a = (self.a - out_byte + in_byte) & MASK
        self.b = (self.b - self.length * out_byte + self.a) & MASK

    def shrink(self, out_byte: int) -> None:
        """Drop ``out_byte`` from the left without adding anything."""
        if self.length == 0:
            raise ValueError("cannot shrink an empty window")
        self.a = (self.a - out_byte) & MASK
        self.b = (self.b - self.length * out_byte) & MASK
        self.length -= 1
