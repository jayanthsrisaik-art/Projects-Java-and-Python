import random

import pytest

from deltasync.bench import make_base
from deltasync.cdc import chunk_hashes, fastcdc_chunks, fixed_chunks


def covers(data, chunks):
    pos = 0
    for o, n in chunks:
        assert o == pos and n > 0
        pos += n
    assert pos == len(data)


def test_chunks_tile_the_input_and_respect_bounds():
    data = make_base(300_000, random.Random(1))
    chunks = fastcdc_chunks(data, 1024, 4096, 16384)
    covers(data, chunks)
    assert all(n <= 16384 for _, n in chunks)
    assert all(n >= 1024 for _, n in chunks[:-1])
    mean = len(data) / len(chunks)
    assert 2048 < mean < 8192   # normalized chunking keeps sizes near the target average


def test_deterministic_and_handles_tiny_inputs():
    data = make_base(50_000, random.Random(2))
    assert fastcdc_chunks(data) == fastcdc_chunks(data)
    assert fastcdc_chunks(b"") == []
    assert fastcdc_chunks(b"abc") == [(0, 3)]
    covers(b"x" * 100_000, fastcdc_chunks(b"x" * 100_000))  # no cut points -> max-size chunks


def test_prefix_insertion_preserves_most_cdc_chunks_but_breaks_fixed():
    old = make_base(200_000, random.Random(3))
    new = b"INSERTED" + old

    def shared(chunker):
        have = set(chunk_hashes(old, chunker(old)))
        nc = chunker(new)
        return sum(n for h, (_, n) in zip(chunk_hashes(new, nc), nc) if h in have) / len(new)

    assert shared(fastcdc_chunks) > 0.9
    assert shared(lambda d: fixed_chunks(d, 8192)) < 0.05


def test_parameter_validation():
    with pytest.raises(ValueError):
        fastcdc_chunks(b"x", 10, 6000, 20000)    # avg not a power of two
    with pytest.raises(ValueError):
        fastcdc_chunks(b"x", 9000, 8192, 20000)  # min > avg
    with pytest.raises(ValueError):
        fixed_chunks(b"x", 0)
