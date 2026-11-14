from asofqa.corpus import TOPICS, generate
from asofqa.evaluate import run_split, summarise
from asofqa.index import Corpus, parse_doc


def test_generation_is_deterministic():
    a = generate(3, "dev")
    b = generate(3, "dev")
    assert a[1] == b[1] and a[2] == b[2]
    assert generate(4, "dev")[1] != a[1]


def test_timeline_invariants():
    for seed in range(20):
        fams, docs, qs = generate(seed, "test")
        for f in fams:
            vals = [v.value for v in f.versions]
            assert len(vals) == len(set(vals)), "values are distinct so answers are unambiguous"
            act = f.active()
            assert act[0].version == 1
            assert sorted(v.effective for v in act) == [v.effective for v in act]
            for w in (v for v in f.versions if v.status == "withdrawn"):
                assert act[0].effective < w.effective
        Corpus([parse_doc(t) for t in docs.values()])  # every generated doc parses
        assert {q["policy"] for q in qs} == {t.key for t in TOPICS}


def test_gold_never_points_at_unapproved_docs():
    fams, docs, qs = generate(11, "dev")
    status = {v.doc_id: v.status for f in fams for v in f.versions}
    for q in qs:
        assert all(status[d] == "active" for d in q["gold_docs"])
        assert q["abstain"] == (not q["gold_docs"])


def test_asof_beats_baselines_on_a_dev_seed():
    s = summarise(run_split([0], "dev"))
    assert s["asof"]["accuracy"] > s["asof_nostatus"]["accuracy"] > s["latest"]["accuracy"] > s["naive"]["accuracy"]
    assert s["asof"]["abstention"]["recall"] == 1.0
    assert s["latest"]["per_qtype"]["previous"]["accuracy"] == 0.0
