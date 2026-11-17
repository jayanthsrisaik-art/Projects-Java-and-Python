"""Chunkers for dedupe-style transfer: fixed-size and FastCDC (Gear hash, normalized chunking).

FastCDC (Xia et al., USENIX ATC 2016): a Gear fingerprint ``fp = (fp << 1) + G[byte]``
is updated per byte; a cut point is declared where ``fp & mask == 0``. Before the
average size a harder mask (more bits) is used, after it an easier one, which pulls
chunk sizes towards the average ("normalized chunking"). The first ``min_size`` bytes
of each chunk are skipped. Because the mask is placed on the high bits, a cut depends
only on the preceding ~64 bytes, so an insertion only disturbs nearby boundaries.
"""
from __future__ import annotations

from hashlib import blake2b

_M64 = (1 << 64) - 1


def _gear_table(seed: bytes = b"deltasync-gear-v1") -> list[int]:
    return [int.from_bytes(blake2b(seed + bytes([i]), digest_size=8).digest(), "little")
            for i in range(256)]


GEAR = _gear_table()


def _high_mask(bits: int) -> int:
    return ((1 << bits) - 1) << (64 - bits)


def fixed_chunks(data: bytes, size: int = 8192) -> list[tuple[int, int]]:
    if size < 1:
        raise ValueError("size must be >= 1")
    return [(i, min(size, len(data) - i)) for i in range(0, len(data), size)]


def fastcdc_chunks(data: bytes, min_size: int = 2048, avg_size: int = 8192,
                   max_size: int = 65536, normalization: int = 2) -> list[tuple[int, int]]:
    if not (0 < min_size <= avg_size <= max_size):
        raise ValueError("need 0 < min_size <= avg_size <= max_size")
    if avg_size & (avg_size - 1):
        raise ValueError("avg_size must be a power of two")
    bits = avg_size.bit_length() - 1
    mask_s = _high_mask(min(63, bits + normalization))
    mask_l = _high_mask(max(1, bits - normalization))
    gear = GEAR
    n = len(data)
    chunks = []
    start = 0
    while start < n:
        remaining = n - start
        if remaining <= min_size:
            chunks.append((start, remaining))
            break
        end = start + min(max_size, remaining)
        normal = start + min(avg_size, remaining)
        fp = 0
        i = start + min_size
        cut = end
        while i < normal:
            fp = ((fp << 1) + gear[data[i]]) & _M64
            i += 1
            if not fp & mask_s:
                cut = i
                break
        else:
            while i < end:
                fp = ((fp << 1) + gear[data[i]]) & _M64
                i += 1
                if not fp & mask_l:
                    cut = i
                    break
        chunks.append((start, cut - start))
        start = cut
    return chunks


def chunk_hashes(data: bytes, chunks: list[tuple[int, int]], digest_size: int = 16) -> list[bytes]:
    return [blake2b(data[o:o + n], digest_size=digest_size).digest() for o, n in chunks]
