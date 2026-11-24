import itertools
import random

import pytest

from rgatext import Replica, oracle_text
from rgatext.sim import fuzz, run_scenario


def concurrent_ops():
    """Three replicas edit the same base text concurrently."""
    base = Replica("base")
    base_ops = base.insert(0, "abc")
    reps = {n: Replica(n) for n in "xyz"}
    for r in reps.values():
        r.receive(base_ops)
    ops = []
    ops += reps["x"].insert(1, "12")       # between a and b
    ops += reps["y"].insert(1, "Q")        # same gap, concurrently
    ops += reps["y"].delete(0, 1)          # y deletes 'a'
    ops += reps["z"].delete(1, 1)          # z deletes 'b'
    ops += reps["z"].insert(2, "!")        # after c
    return base_ops, ops


def test_every_delivery_order_yields_the_same_text():
    base_ops, ops = concurrent_ops()
    assert len(ops) == 6
    texts = set()
    for perm in itertools.permutations(ops):
        r = Replica("obs")
        r.receive(base_ops)
        r.receive(perm)
        assert r.pending_count() == 0
        texts.add(r.text())
    assert len(texts) == 1
    assert texts == {oracle_text(base_ops + ops)}


@pytest.mark.parametrize("seed", range(20))
def test_random_order_with_duplicates_and_split_batches_matches_oracle(seed):
    rng = random.Random(seed)
    base_ops, ops = concurrent_ops()
    stream = base_ops + ops + rng.sample(ops, k=3)
    rng.shuffle(stream)
    r = Replica("obs")
    cut = rng.randrange(len(stream))
    r.receive(stream[:cut])
    r.receive(stream[cut:])
    assert r.pending_count() == 0
    assert r.text() == oracle_text(base_ops + ops)


@pytest.mark.parametrize("seed", range(10))
def test_fuzzed_multi_replica_sessions_converge_to_oracle(seed):
    result = fuzz(replicas=4, steps=250, seed=seed)
    assert result["crdt_converged"]
    assert result["crdt_matches_oracle"]
    assert result["crdt_pending"] == 0


def test_naive_index_replay_baseline_diverges_where_crdt_converges():
    scenario = {
        "replicas": ["a", "b"],
        "steps": [
            {"edit": "a", "insert": [0, "abcd"]},
            {"sync": ["a", "b"]},
            {"edit": "a", "insert": [1, "X"]},
            {"edit": "b", "delete": [0, 2]},
            {"sync": ["a", "b"]},
        ],
    }
    result = run_scenario(scenario)
    assert result["crdt"]["converged"]
    assert result["crdt"]["texts"]["a"] == "Xcd"
    assert not result["naive"]["converged"]
