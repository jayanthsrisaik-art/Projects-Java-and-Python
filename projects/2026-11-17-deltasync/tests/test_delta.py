import random
from collections import defaultdict

import pytest

from deltasync import (Copy, Delta, DeltaError, FormatError, Literal, Signature, apply_delta,
                       default_block_size, generate_delta, make_signature)
from deltasync.rollsum import weak_checksum
from deltasync.signature import strong_hash


def roundtrip(old: bytes, new: bytes, bs=None, strong_len=8) -> Delta:
    sig = Signature.from_bytes(make_signature(old, bs, strong_len).to_bytes())
    delta = Delta.from_bytes(generate_delta(sig, new).to_bytes())
    assert apply_delta(old, delta) == new
    return delta


def random_edit(rng: random.Random, data: bytes) -> bytes:
    buf = bytearray(data)
    for _ in range(rng.randint(0, 6)):
        kind = rng.choice(["ins", "del", "sub", "move", "dup"])
        p = rng.randint(0, len(buf))
        ln = rng.randint(0, 200)
        if kind == "ins":
            buf[p:p] = bytes(rng.getrandbits(8) for _ in range(ln))
        elif kind == "del":
            del buf[p:p + ln]
        elif kind == "sub":
            buf[p:p + ln] = bytes(rng.getrandbits(8) for _ in range(min(ln, len(buf) - p)))
        elif kind == "move":
            seg = bytes(buf[p:p + ln]); del buf[p:p + ln]
            q = rng.randint(0, len(buf)); buf[q:q] = seg
        else:
            buf[p:p] = buf[p:p + ln]
    return bytes(buf)


def test_randomized_roundtrips_many_shapes():
    rng = random.Random(42)
    for case in range(300):
        size = rng.choice([0, 1, 5, 63, 64, 65, 500, 2000, 5000])
        alphabet = rng.choice([2, 4, 256])  # low-entropy alphabets create many weak collisions
        old = bytes(rng.randrange(alphabet) for _ in range(size))
        new = random_edit(rng, old)
        bs = rng.choice([1, 3, 16, 64, 100])
        roundtrip(old, new, bs, strong_len=rng.choice([4, 8, 16]))


@pytest.mark.parametrize("old,new", [(b"", b""), (b"", b"abc"), (b"abc", b""),
                                     (b"x", b"x"), (b"short", b"shorter")])
def test_edge_cases(old, new):
    roundtrip(old, new, bs=4)


def test_identical_file_is_one_copy():
    data = bytes(range(256)) * 40
    d = roundtrip(data, data, bs=100)
    assert d.ops == [Copy(0, len(data))]
    assert d.literal_bytes == 0


def test_shifted_data_still_matches_every_block():
    rng = random.Random(1)
    old = bytes(rng.getrandbits(8) for _ in range(10_000))
    new = b"PREFIX" + old
    d = roundtrip(old, new, bs=500)
    assert d.literal_bytes == 6
    assert d.ops == [Literal(b"PREFIX"), Copy(0, 10_000)]


def test_short_tail_block_matches_only_at_end_of_target():
    rng = random.Random(2)
    old = bytes(rng.getrandbits(8) for _ in range(1050))  # blocks: 500, 500, 50 (tail)
    assert roundtrip(old, old + b"", bs=500).literal_bytes == 0
    d = roundtrip(old, old + b"!", bs=500)
    assert d.literal_bytes == 51  # tail no longer at end: sent literally (rsync behaviour)


def test_moved_and_duplicated_blocks_are_found():
    rng = random.Random(3)
    a, b, c = (bytes(rng.getrandbits(8) for _ in range(256)) for _ in range(3))
    d = roundtrip(a + b + c, c + a + a + b, bs=256)
    assert d.literal_bytes == 0
    assert d.ops == [Copy(512, 256), Copy(0, 256), Copy(0, 512)]


def _weak_collision(bs: int):
    rng = random.Random(5)
    seen = {}
    while True:
        blk = bytes(rng.getrandbits(8) for _ in range(bs))
        w = weak_checksum(blk)
        if w in seen and seen[w] != blk:
            return seen[w], blk
        seen[w] = blk


def test_weak_collision_rejected_by_strong_hash():
    x, y = _weak_collision(4)
    assert weak_checksum(x) == weak_checksum(y) and x != y
    d = generate_delta(make_signature(x, 4), y)
    assert apply_delta(x, d) == y
    assert d.stats.weak_hits == 1 and d.stats.false_weak_hits == 1
    assert d.copy_bytes == 0


def test_strong_collision_is_caught_by_whole_file_hash():
    # with a 1-byte strong hash, find two different 4-byte blocks equal in weak AND strong
    rng = random.Random(9)
    seen = {}
    while True:
        blk = bytes(rng.getrandbits(8) for _ in range(4))
        key = (weak_checksum(blk), strong_hash(blk, 1))
        if key in seen and seen[key] != blk:
            x, y = seen[key], blk
            break
        seen[key] = blk
    delta = generate_delta(make_signature(x, 4, 1), y)
    assert delta.ops == [Copy(0, 4)]          # fooled into a wrong copy...
    with pytest.raises(DeltaError, match="hash mismatch"):
        apply_delta(x, delta)                 # ...but verification refuses the output


def test_patch_against_wrong_basis_fails():
    rng = random.Random(4)
    old = bytes(rng.getrandbits(8) for _ in range(4000))
    delta = generate_delta(make_signature(old, 256), old[:2000] + b"zz" + old[2000:])
    with pytest.raises(DeltaError):
        apply_delta(old[:1000], delta)          # copy out of range
    tampered = bytearray(old); tampered[10] ^= 1
    with pytest.raises(DeltaError, match="hash mismatch"):
        apply_delta(bytes(tampered), delta)


def test_corrupt_delta_wire_rejected():
    old = b"hello world " * 100
    wire = generate_delta(make_signature(old, 32), old + b"tail").to_bytes()
    with pytest.raises(FormatError):
        Delta.from_bytes(b"XXXX" + wire[4:])
    with pytest.raises(FormatError):
        Delta.from_bytes(wire[:-1])            # missing END
    with pytest.raises(FormatError):
        Delta.from_bytes(wire + b"\x00")       # trailing garbage
    bad_tag = bytearray(wire); bad_tag[4 + 1 + 2 + 32] = 0x7F
    with pytest.raises(FormatError, match="unknown op"):
        Delta.from_bytes(bytes(bad_tag))


def test_signature_wire_validation():
    sig = make_signature(b"a" * 1000, 100)
    wire = sig.to_bytes()
    assert Signature.from_bytes(wire) == sig
    with pytest.raises(FormatError):
        Signature.from_bytes(wire[:-1])
    with pytest.raises(FormatError):
        make_signature(b"abc", 0)
    with pytest.raises(FormatError):
        make_signature(b"abc", 10, 33)


def test_default_block_size_heuristic():
    assert default_block_size(0) == 700
    assert default_block_size(490_000) == 700
    assert default_block_size(10**8) == 10000
    assert default_block_size(10**12) == 128 * 1024
    assert default_block_size(10**8) % 8 == 0


def test_delta_much_smaller_than_file_for_small_edit():
    rng = random.Random(6)
    old = bytes(rng.getrandbits(8) for _ in range(200_000))
    new = old[:100_000] + b"EDIT" + old[100_004:]
    sig = make_signature(old)
    delta = generate_delta(sig, new)
    assert apply_delta(old, delta) == new
    assert len(sig.to_bytes()) + len(delta.to_bytes()) < 0.05 * len(new)
