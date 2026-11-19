import datetime as dt

import pytest

from safeshare.pseudonym import DEMO_KEY, KeyError_, Pseudonymizer, load_key

KEY = b"k" * 32


def test_tokens_stable_domain_separated_and_keyed():
    a = Pseudonymizer(KEY)
    assert a.token("patient", "P1") == Pseudonymizer(KEY).token("patient", "P1")
    assert a.token("patient", "P1") != a.token("encounter", "P1")
    assert a.token("patient", "P1") != a.token("patient", "P2")
    assert a.token("patient", "P1") != Pseudonymizer(b"j" * 32).token("patient", "P1")
    assert a.token("patient", "P1").startswith("pat_")


def test_tokens_do_not_collide_on_many_ids():
    pz = Pseudonymizer(KEY)
    toks = {pz.token("patient", f"P{i:06d}") for i in range(20000)}
    assert len(toks) == 20000


def test_shift_is_bounded_nonzero_and_preserves_intervals():
    pz = Pseudonymizer(KEY, max_shift_days=30)
    offs = {pz.shift_days("patient", f"P{i}") for i in range(2000)}
    assert 0 not in offs
    assert min(offs) >= -30 and max(offs) <= 30
    assert len(offs) > 50
    d1 = pz.shift_date("patient", "P7", "2024-02-28")
    d2 = pz.shift_date("patient", "P7", "2024-03-05")
    gap = dt.date.fromisoformat(d2) - dt.date.fromisoformat(d1)
    assert gap.days == 6


def test_shift_keeps_time_suffix_and_blank():
    pz = Pseudonymizer(KEY)
    out = pz.shift_date("patient", "P1", "2024-01-01T08:30:00")
    assert out.endswith("T08:30:00")
    assert pz.shift_date("patient", "P1", "") == ""


def test_key_loading(tmp_path, monkeypatch):
    monkeypatch.delenv("SAFESHARE_KEY", raising=False)
    with pytest.raises(KeyError_):
        load_key()
    f = tmp_path / "key"
    f.write_bytes(b"short\n")
    with pytest.raises(KeyError_):
        load_key(str(f))
    f.write_bytes(b"0123456789abcdef0123\n")
    assert load_key(str(f)) == b"0123456789abcdef0123"
    monkeypatch.setenv("SAFESHARE_KEY", "an-environment-provided-key")
    assert load_key() == b"an-environment-provided-key"
    assert load_key(demo=True) == DEMO_KEY
