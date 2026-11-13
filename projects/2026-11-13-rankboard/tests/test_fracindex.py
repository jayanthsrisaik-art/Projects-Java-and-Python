import json
import random
from pathlib import Path

import pytest

from rankboard.fracindex import (
    DIGITS, KeyError_, fraction_only_between, is_valid_key, key_between, keys_sequence,
    validate_key,
)

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def test_alphabet_is_ascii_sorted():
    # string comparison must equal key order in Python, JS and SQLite BINARY collation
    assert list(DIGITS) == sorted(DIGITS)
    assert len(DIGITS) == 62


def test_first_keys():
    assert key_between(None, None) == "a0"
    assert key_between("a0", None) == "a1"
    assert key_between(None, "a0") == "Zz"
    assert key_between("a0", "a1") == "a0V"
    assert keys_sequence(3) == ["a0", "a1", "a2"]


@pytest.mark.parametrize("seed", range(40))
def test_random_insertions_keep_intended_order(seed):
    """Oracle: a Python list holds the intended order; keys must sort the same way."""
    rng = random.Random(seed)
    keys: list[str] = []
    for _ in range(400):
        i = rng.randint(0, len(keys))
        a = keys[i - 1] if i > 0 else None
        b = keys[i] if i < len(keys) else None
        k = key_between(a, b)
        validate_key(k)
        assert (a is None or a < k) and (b is None or k < b)
        keys.insert(i, k)
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)


def test_append_and_prepend_stay_short():
    k = None
    for _ in range(10_000):
        k = key_between(k, None)
    assert len(k) <= 4
    k = None
    for _ in range(10_000):
        k = key_between(None, k)
    assert len(k) <= 4


def test_fraction_only_baseline_grows_on_append():
    k = None
    for _ in range(600):
        k = fraction_only_between(k, None)
    assert len(k) >= 90  # ~1 char per 6 appends: why the integer part exists


def test_hotspot_growth_is_logarithmic_per_insert():
    a, b = "a0", "a1"
    for _ in range(120):
        b = key_between(a, b)
    # each insert halves the gap: ~log2(62) = 5.95 inserts per extra character
    assert 18 <= len(b) <= 25


def test_integer_length_class_transitions():
    assert key_between("az", None) == "b00"
    assert key_between(None, "Z0") == "Yzz"
    assert "Zz" < key_between("Zz", "a0") < "a0"
    top = "z" + "z" * 26
    assert key_between(top, None) > top  # falls back to a fraction once integers run out


def test_smallest_integer_is_never_returned_bare():
    k = key_between(None, "A00000000000000000000000001")
    assert is_valid_key(k) and k < "A00000000000000000000000001"
    assert key_between(None, k) < k


@pytest.mark.parametrize("a,b", [("a0", "a0"), ("a1", "a0"), ("a00", None), ("b0", None),
                                 ("a$", None), ("", None), ("A" + "0" * 26, None)])
def test_invalid_inputs_raise(a, b):
    with pytest.raises(KeyError_):
        key_between(a, b)


def test_vectors_file_is_current():
    data = json.loads((SAMPLES / "key_vectors.json").read_text())
    assert len(data["between"]) > 200
    for a, b, expected in data["between"]:
        assert key_between(a, b) == expected
    for a, b in data["invalid"]:
        with pytest.raises(KeyError_):
            key_between(a, b)
    assert keys_sequence(10) == data["sequence_10"]


def test_vectors_file_is_regenerated_from_code():
    from rankboard.vectors import build
    assert json.loads((SAMPLES / "key_vectors.json").read_text()) == build()
