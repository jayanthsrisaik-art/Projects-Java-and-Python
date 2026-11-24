import random

import pytest

from rgatext import CRDTError, Replica, oracle_text, validate_op
from rgatext.doc import BLOCK_MAX


def sync(a, b):
    b.receive(a.ops_since(b.version()))
    a.receive(b.ops_since(a.version()))


def test_local_insert_and_delete_behave_like_a_string():
    r = Replica("a")
    r.insert(0, "hello world")
    r.delete(5, 6)
    r.insert(5, ", there")
    r.insert(0, ">> ")
    assert r.text() == ">> hello, there"
    assert len(r) == len(r.text())


def test_single_replica_matches_python_string_model_across_block_splits():
    rng = random.Random(42)
    r, model = Replica("solo"), ""
    for _ in range(3000):
        if model and rng.random() < 0.3:
            pos = rng.randrange(len(model))
            n = rng.randint(1, min(5, len(model) - pos))
            r.delete(pos, n)
            model = model[:pos] + model[pos + n:]
        else:
            pos = rng.randint(0, len(model))
            s = "".join(rng.choice("xyz") for _ in range(rng.randint(1, 3)))
            r.insert(pos, s)
            model = model[:pos] + s + model[pos:]
    assert r.text() == model
    assert r.stats()["blocks"] > 1  # the chunked index really was exercised
    assert len(r) == len(model)
    assert oracle_text(r.snapshot_ops()) == model


@pytest.mark.parametrize("pos,text", [(-1, "x"), (1, "x"), ("0", "x"), (0, 5)])
def test_insert_rejects_bad_arguments(pos, text):
    with pytest.raises(CRDTError):
        Replica("a").insert(pos, text)


@pytest.mark.parametrize("pos,length", [(0, 4), (3, 1), (-1, 1), (0, -1)])
def test_delete_rejects_out_of_range(pos, length):
    r = Replica("a")
    r.insert(0, "abc")
    with pytest.raises(CRDTError):
        r.delete(pos, length)


def test_concurrent_runs_at_same_position_do_not_interleave():
    a, b = Replica("a"), Replica("b")
    a.insert(0, "[]")
    sync(a, b)
    a.insert(1, "alpha")
    b.insert(1, "BETA")
    sync(a, b)
    assert a.text() == b.text()
    assert a.text() in ("[alphaBETA]", "[BETAalpha]")


def test_insert_survives_concurrent_delete_of_its_neighbour():
    a, b = Replica("a"), Replica("b")
    a.insert(0, "cat")
    sync(a, b)
    a.delete(1, 1)           # a deletes 'a'
    b.insert(2, "!")         # b inserts after the 'a' a just deleted
    sync(a, b)
    assert a.text() == b.text() == "c!t"


def test_concurrent_deletes_of_same_char_are_idempotent():
    a, b = Replica("a"), Replica("b")
    a.insert(0, "abc")
    sync(a, b)
    a.delete(1, 1)
    b.delete(1, 1)
    sync(a, b)
    assert a.text() == b.text() == "ac"
    assert len(a) == len(b) == 2
    assert a.stats()["tombstones"] == 1


def test_out_of_order_ops_are_buffered_until_dependencies_arrive():
    a, b = Replica("a"), Replica("b")
    ops = a.insert(0, "abc") + a.delete(0, 1)
    report = b.receive(list(reversed(ops))[:2])   # delete + 'c' only
    assert report.applied == 0 and report.buffered == 2
    assert b.text() == ""
    report = b.receive(ops)                        # everything, incl. repeats
    assert report.duplicates == 2 and report.buffered == 0
    assert b.text() == "bc"


def test_duplicates_are_ignored():
    a, b = Replica("a"), Replica("b")
    ops = a.insert(0, "hi")
    b.receive(ops)
    report = b.receive(ops + ops)
    assert report.applied == 0 and report.duplicates == 4
    assert b.text() == "hi"


def test_delta_sync_sends_only_missing_ops():
    a, b = Replica("a"), Replica("b")
    a.insert(0, "abc")
    sync(a, b)
    assert a.ops_since(b.version()) == []
    a.insert(3, "d")
    delta = a.ops_since(b.version())
    assert [op["ch"] for op in delta] == ["d"]
    assert b.version() == {"a": 3}


def test_lamport_clock_advances_past_remote_ops():
    a, b = Replica("a"), Replica("b")
    a.insert(0, "xxxxx")
    b.receive(a.ops_since())
    op = b.insert(0, "y")[0]
    assert op["id"][0] == 6  # newer than everything b has seen


@pytest.mark.parametrize("bad", [
    "not a dict",
    {"kind": "ins", "rep": "a", "seq": 0, "id": [1, "a"], "origin": [0, ""], "ch": "x"},
    {"kind": "ins", "rep": "a", "seq": 1, "id": [1, "b"], "origin": [0, ""], "ch": "x"},
    {"kind": "ins", "rep": "a", "seq": 1, "id": [1, "a"], "origin": [0, ""], "ch": "xy"},
    {"kind": "ins", "rep": "a", "seq": 1, "id": [3, "a"], "origin": [3, "b"], "ch": "x"},
    {"kind": "ins", "rep": "a", "seq": 1, "id": ["1", "a"], "origin": [0, ""], "ch": "x"},
    {"kind": "ins", "rep": "a", "seq": True, "id": [1, "a"], "origin": [0, ""], "ch": "x"},
    {"kind": "del", "rep": "a", "seq": 1, "target": [0, ""]},
    {"kind": "upd", "rep": "a", "seq": 1},
    {"kind": "del", "rep": "", "seq": 1, "target": [1, "a"]},
])
def test_validate_op_rejects_malformed_operations(bad):
    with pytest.raises(CRDTError):
        validate_op(bad)
    with pytest.raises(CRDTError):
        Replica("z").receive([bad])


def test_reusing_a_replica_id_is_detected():
    a1, a2, c = Replica("a"), Replica("a"), Replica("c")
    c.receive(a1.insert(0, "x"))
    other = a2.insert(0, "y")[0]
    other["seq"] = 2  # a second device wrongly sharing id "a"
    with pytest.raises(CRDTError, match="already exists"):
        c.receive([other])


def test_anchor_tracks_cursor_through_remote_edits():
    a, b = Replica("a"), Replica("b")
    a.insert(0, "hello world")
    sync(a, b)
    cursor = a.anchor(6)            # just before "world"
    b.insert(0, ">>> ")             # remote insert before the cursor
    b.insert(len(b), "!")           # and after it
    sync(a, b)
    assert a.text() == ">>> hello world!"
    assert a.resolve(cursor) == 10
    assert a.text()[a.resolve(cursor):].startswith("world")


def test_anchor_on_deleted_char_falls_back_to_left_neighbour():
    r = Replica("a")
    r.insert(0, "abcdef")
    cursor = r.anchor(4)            # after 'd'
    r.delete(2, 2)                  # delete 'cd'
    assert r.text() == "abef"
    assert r.resolve(cursor) == 2   # gap between 'b' and 'e'
    assert r.resolve(r.anchor(0)) == 0
    with pytest.raises(CRDTError):
        r.resolve([99, "nobody"])


def test_rebuild_from_op_log_restores_state_and_counters():
    a = Replica("a")
    a.insert(0, "persist me")
    a.delete(0, 8)
    restored = Replica.from_ops("a", a.snapshot_ops())
    assert restored.text() == "me"
    assert restored.version() == a.version()
    # the restored replica keeps issuing fresh, non-colliding ids
    op = restored.insert(0, "x")[0]
    assert op["seq"] == a.version()["a"] + 1
    assert op["id"][0] > max(o["id"][0] for o in a.snapshot_ops() if o["kind"] == "ins")


def test_rebuild_from_incomplete_log_fails_loudly():
    a = Replica("a")
    ops = a.insert(0, "abc")
    with pytest.raises(CRDTError, match="missing dependencies"):
        Replica.from_ops("a", ops[1:])


def test_large_document_spans_many_blocks_and_stays_consistent():
    a, b = Replica("a"), Replica("b")
    a.insert(0, "x" * (BLOCK_MAX * 5))
    b.receive(reversed(a.ops_since()))
    a.insert(BLOCK_MAX * 2, "MID")
    b.delete(BLOCK_MAX * 4, 10)
    sync(a, b)
    assert a.text() == b.text()
    assert "MID" in a.text() and len(a) == BLOCK_MAX * 5 + 3 - 10
    assert a.stats()["blocks"] >= 5
