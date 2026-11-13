import random
import threading

import pytest

from rankboard.store import (
    Conflict, FeedGone, Invalid, NotFound, Store, VersionMismatch, replay_feed,
)


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "t.db"))


def make_board(store, cols=("Todo", "Doing", "Done")):
    snap = store.create_board("b", list(cols))
    return snap["board"]["id"], [c["id"] for c in snap["columns"]]


def order(store, board, col):
    return [c["id"] for c in store.snapshot(board)["cards"] if c["column_id"] == col]


def test_append_and_anchored_placement(store):
    board, (todo, *_rest) = make_board(store)
    for cid in ("a", "b", "c"):
        store.create_card(board, todo, cid, card_id=cid)
    _, info = store.create_card(board, todo, "x", after_id="a", before_id="b", card_id="x")
    assert info["placement"] == "after"
    _, info = store.create_card(board, todo, "y", before_id="a", card_id="y")
    assert info["placement"] == "before"
    assert order(store, board, todo) == ["y", "a", "x", "b", "c"]


def test_concurrent_drop_into_same_gap_both_land_between_anchors(store):
    board, (todo, *_rest) = make_board(store)
    store.create_card(board, todo, "a", card_id="a")
    store.create_card(board, todo, "b", card_id="b")
    # both clients saw [a, b] and dropped between them
    store.create_card(board, todo, "p", after_id="a", before_id="b", card_id="p")
    store.create_card(board, todo, "q", after_id="a", before_id="b", card_id="q")
    assert order(store, board, todo) == ["a", "q", "p", "b"]


def test_stale_anchor_falls_back(store):
    board, (todo, doing, _done) = make_board(store)
    for cid in ("a", "b", "c"):
        store.create_card(board, todo, cid, card_id=cid)
    store.move_card("a", doing)  # someone moved the anchor away
    card, info = store.move_card("c", todo, after_id="a", before_id="b")
    assert info["placement"] == "before"  # after-anchor gone, before-anchor still valid
    assert order(store, board, todo) == ["c", "b"]
    _, info = store.move_card("c", todo, after_id="a", before_id="zzz")
    assert info["placement"] == "fallback_end"
    assert order(store, board, todo) == ["b", "c"]


def test_move_relative_to_itself_rejected(store):
    board, (todo, *_rest) = make_board(store)
    store.create_card(board, todo, "a", card_id="a")
    with pytest.raises(Invalid):
        store.move_card("a", todo, after_id="a")


def test_move_within_column_ignores_its_own_old_slot(store):
    board, (todo, *_rest) = make_board(store)
    for cid in "abcd":
        store.create_card(board, todo, cid, card_id=cid)
    store.move_card("a", todo, after_id="c", before_id="d")
    assert order(store, board, todo) == ["b", "c", "a", "d"]
    store.move_card("d", todo, before_id="b")
    assert order(store, board, todo) == ["d", "b", "c", "a"]


def test_column_must_belong_to_board(store):
    b1, (c1, *_r) = make_board(store)
    b2, (c2, *_r2) = make_board(store)
    with pytest.raises(Invalid):
        store.create_card(b1, c2, "x")
    store.create_card(b1, c1, "x", card_id="x")
    with pytest.raises(Invalid):
        store.move_card("x", c2)


def test_client_id_makes_create_idempotent(store):
    board, (todo, *_rest) = make_board(store)
    first, info1 = store.create_card(board, todo, "Write tests", card_id="k1")
    again, info2 = store.create_card(board, todo, "Write tests", card_id="k1")
    assert info1["created"] and not info2["created"]
    assert first == again
    assert len(store.snapshot(board)["cards"]) == 1
    assert len(store.changes(board, 0)["changes"]) == 1  # replay is not a new change
    with pytest.raises(Conflict):
        store.create_card(board, todo, "Something else", card_id="k1")
    store.delete_card("k1", if_version=1)
    with pytest.raises(Conflict):  # a deleted id cannot be resurrected by a late retry
        store.create_card(board, todo, "Write tests", card_id="k1")


def test_bad_ids_and_titles(store):
    board, (todo, *_rest) = make_board(store)
    with pytest.raises(Invalid):
        store.create_card(board, todo, "x", card_id="has space")
    with pytest.raises(Invalid):
        store.create_card(board, todo, "   ")
    with pytest.raises(NotFound):
        store.create_card("nope", todo, "x")
    with pytest.raises(NotFound):
        store.move_card("nope", todo)


def test_edit_requires_current_version(store):
    board, (todo, *_rest) = make_board(store)
    store.create_card(board, todo, "v1", card_id="c")
    assert store.edit_card("c", 1, title="v2")["version"] == 2
    with pytest.raises(VersionMismatch) as exc:
        store.edit_card("c", 1, title="lost update")
    assert exc.value.current["title"] == "v2"
    with pytest.raises(VersionMismatch):
        store.delete_card("c", 1)
    # an optional version on move is checked when given
    with pytest.raises(VersionMismatch):
        store.move_card("c", todo, if_version=1)
    assert store.move_card("c", todo, if_version=2)[0]["version"] == 3


def test_rebalance_preserves_order_and_versions(tmp_path):
    store = Store(str(tmp_path / "r.db"), max_key_len=4)
    board, (todo, *_rest) = make_board(store)
    store.create_card(board, todo, "first", card_id="first")
    store.create_card(board, todo, "last", card_id="last")
    expected = ["first", "last"]
    rebalances = 0
    for i in range(40):  # hotspot: always right after "first"
        cid = f"h{i}"
        _, info = store.create_card(board, todo, cid, after_id="first", card_id=cid)
        rebalances += info["rebalanced"] > 0
        expected.insert(1, cid)
        assert order(store, board, todo) == expected
    assert rebalances >= 2
    assert store.check_invariants(board) == []
    cards = store.snapshot(board)["cards"]
    assert all(len(c["rank"]) <= 4 for c in cards)
    assert all(c["version"] == 1 for c in cards)  # rank rewrite is not a content change
    reasons = {ch["reason"] for ch in store.changes(board, 0, limit=1000)["changes"]}
    assert "rebalance" in reasons


def test_feed_paging_and_replay_converges(store):
    board, cols = make_board(store)
    rng = random.Random(3)
    ids: list[str] = []
    for i in range(120):
        r = rng.random()
        if not ids or r < 0.5:
            cid = f"c{i}"
            store.create_card(board, rng.choice(cols), cid, card_id=cid)
            ids.append(cid)
        elif r < 0.8:
            cid = rng.choice(ids)
            col = rng.choice(cols)
            peers = [c for c in order(store, board, col) if c != cid]
            store.move_card(cid, col, after_id=rng.choice(peers + [None]))
        elif r < 0.9:
            cid = rng.choice(ids)
            v = store.get_card(cid)["version"]
            store.edit_card(cid, v, title=f"t{i}")
        else:
            cid = ids.pop(rng.randrange(len(ids)))
            store.delete_card(cid, store.get_card(cid)["version"])
    state: dict = {}
    cursor, pages = 0, 0
    while True:
        page = store.changes(board, cursor, limit=17)
        state = replay_feed(page["changes"], state)
        assert page["cursor"] >= cursor
        cursor, pages = page["cursor"], pages + 1
        if not page["has_more"]:
            break
    assert pages > 1
    snap = store.snapshot(board)
    assert state == {c["id"]: c for c in snap["cards"]}
    assert cursor == snap["seq"]
    assert store.changes(board, cursor)["changes"] == []


def test_compaction_returns_gone_for_stale_cursors(store):
    board, (todo, *_rest) = make_board(store)
    for i in range(10):
        store.create_card(board, todo, f"c{i}", card_id=f"c{i}")
    head = store.snapshot(board)["seq"]
    assert store.compact(board, keep_last=3) == 7
    with pytest.raises(FeedGone):
        store.changes(board, 0)
    tail = store.changes(board, head - 3)
    assert len(tail["changes"]) == 3
    # a fresh snapshot gives a cursor that works
    snap = store.snapshot(board)
    assert store.changes(board, snap["seq"])["changes"] == []


def test_threads_dropping_into_same_gap(tmp_path):
    """Real concurrency: 8 threads x 15 inserts after the same anchor card."""
    store = Store(str(tmp_path / "c.db"), max_key_len=12)
    board, (todo, *_rest) = make_board(store)
    store.create_card(board, todo, "anchor", card_id="anchor")
    store.create_card(board, todo, "tail", card_id="tail")
    errors: list[BaseException] = []

    def worker(n):
        try:
            for i in range(15):
                store.create_card(board, todo, f"w{n}-{i}", after_id="anchor",
                                  before_id="tail", card_id=f"w{n}-{i}")
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    ids = order(store, board, todo)
    assert len(ids) == 2 + 8 * 15
    assert ids[0] == "anchor" and ids[-1] == "tail"
    # each worker's own inserts appear newest-first (each went right after the anchor)
    for n in range(8):
        mine = [c for c in ids if c.startswith(f"w{n}-")]
        assert mine == [f"w{n}-{i}" for i in reversed(range(15))]
    assert store.check_invariants(board) == []
