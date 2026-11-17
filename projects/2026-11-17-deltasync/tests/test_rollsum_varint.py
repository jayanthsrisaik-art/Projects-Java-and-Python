import random

import pytest

from deltasync import varint
from deltasync.errors import FormatError
from deltasync.rollsum import RollingChecksum, weak_checksum


def test_rolling_matches_direct_at_every_offset():
    rng = random.Random(0)
    data = bytes(rng.getrandbits(8) for _ in range(3000))
    for L in (1, 7, 64, 700):
        rc = RollingChecksum(data[:L])
        for k in range(len(data) - L):
            assert rc.digest == weak_checksum(data[k:k + L])
            rc.roll(data[k], data[k + L])


def test_shrink_matches_direct():
    data = bytes(range(200, 256)) * 3
    rc = RollingChecksum(data)
    for k in range(len(data)):
        assert rc.digest == weak_checksum(data[k:])
        rc.shrink(data[k])
    assert rc.digest == weak_checksum(b"") == 0
    with pytest.raises(ValueError):
        rc.shrink(0)


def test_weak_checksum_is_position_sensitive():
    assert weak_checksum(b"ab") != weak_checksum(b"ba")


@pytest.mark.parametrize("v", [0, 1, 127, 128, 300, 2**32, 2**63 - 1])
def test_varint_roundtrip(v):
    enc = varint.encode(v)
    assert varint.decode(enc, 0) == (v, len(enc))


def test_varint_rejects_bad_input():
    with pytest.raises(FormatError):
        varint.decode(b"\x80", 0)          # truncated
    with pytest.raises(FormatError):
        varint.decode(b"\x81\x00", 0)      # overlong (1 encoded in 2 bytes)
    with pytest.raises(FormatError):
        varint.decode(b"\xff" * 11, 0)     # too long
    with pytest.raises(ValueError):
        varint.encode(-1)
