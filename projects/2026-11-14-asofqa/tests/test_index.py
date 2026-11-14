import pytest

from asofqa.index import BM25, Corpus, chunk_doc, parse_doc
from tests.helpers import PTO, doc, mini_corpus


def test_parse_doc_reads_front_matter():
    d = PTO[1]
    assert (d.doc_id, d.status, d.version, d.effective.isoformat()) == ("pto-v2", "withdrawn", 2, "2022-03-01")


@pytest.mark.parametrize("bad", [
    "no front matter",
    "---\ndoc_id: x\n---\nbody",
    "---\ndoc_id: x\npolicy: p\ntitle: t\nversion: 1\nstatus: archived\neffective: 2020-01-01\n---\nbody",
])
def test_parse_doc_rejects_bad_headers(bad):
    with pytest.raises(ValueError):
        parse_doc(bad)


def test_chunks_carry_title_and_drop_heading():
    chunks = chunk_doc(PTO[2])
    assert len(chunks) == 2
    assert all(c.text.startswith("Paid Time Off Policy. ") for c in chunks)
    assert "#" not in chunks[0].text


def test_bm25_ranks_the_relevant_family_first():
    c = mini_corpus()
    top = c.bm25.search("daily meal allowance", 3)
    assert top[0][0].doc.policy == "meal"
    assert BM25([]).search("anything") == []


def test_family_sorted_by_effective_date():
    c = mini_corpus()
    assert [d.doc_id for d in c.family["pto"]] == ["pto-v1", "pto-v2", "pto-v3", "pto-v4", "pto-v5"]


def test_from_dir(tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ndoc_id: a\npolicy: p\ntitle: T\nversion: 1\nstatus: active\neffective: 2020-01-01\n---\nBody 5 days.\n")
    assert list(Corpus.from_dir(tmp_path).docs) == ["a"]
