import json
from pathlib import Path

from clickrank.catalog import build_catalog, vocabulary
from clickrank.cli import main
from clickrank.features import SearchIndex, rank
from clickrank.train import load_model


def test_sample_model_reproduces_expected_rankings(samples):
    catalog = build_catalog(0)
    index = SearchIndex(catalog.products, vocabulary())
    w, meta = load_model(samples / "sample_model.json")
    assert meta["method"] == "ipw_estimated"
    expected = json.loads((samples / "expected_rankings.json").read_text())
    queries = [q for q in (samples / "queries.txt").read_text().splitlines() if q]
    assert list(expected) == queries
    for q in queries:
        assert rank(index.candidates(q), w)[:5] == expected[q], q


def test_cli_simulate_ingest_train_search(tmp_path: Path, capsys):
    art, events, db = tmp_path / "art", tmp_path / "w" / "events.jsonl", tmp_path / "w" / "logs.db"
    assert main(["simulate", "--seed", "4", "--impressions", "6000", "--swap-rate", "0.3",
                 "--artifacts", str(art), "--events", str(events)]) == 0
    assert main(["ingest", str(events), "--db", str(db)]) == 0
    report = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert report["impressions_loaded"] > 5000
    assert main(["train", "--db", str(db), "--artifacts", str(art)]) == 0
    capsys.readouterr()
    model = json.loads((art / "model.json").read_text())
    assert model["method"] == "ipw_estimated" and set(model["propensity"]) == {str(k) for k in range(1, 11)}
    assert main(["search", "waterproof tent", "--artifacts", str(art), "-k", "3"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 3 and out[0].lstrip().startswith("1. P")
