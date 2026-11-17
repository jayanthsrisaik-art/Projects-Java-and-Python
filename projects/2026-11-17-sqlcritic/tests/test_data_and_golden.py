import hashlib
import sqlite3

from sqlcritic import golden
from sqlcritic.contract import infer_contract
from sqlcritic.critic import Critic
from sqlcritic.db import build_db, descendants
from sqlcritic.execute import run_sql


def _digest(path):
    con = sqlite3.connect(path)
    h = hashlib.sha256()
    for t in ("regions", "customers", "products", "orders", "order_items"):
        for row in con.execute(f"SELECT * FROM {t} ORDER BY 1, 2"):
            h.update(repr(row).encode())
    return h.hexdigest()


def test_database_is_deterministic(tmp_path):
    assert _digest(build_db(tmp_path / "a.db")) == _digest(build_db(tmp_path / "b.db"))


def test_connection_is_read_only(con):
    res = run_sql(con, "DELETE FROM orders")
    assert not res.ok and "readonly" in res.error.replace(" ", "").lower()


def test_golden_set_shape_and_splits():
    exs = golden.generate()
    assert len(exs) == 352 and len({e.id for e in exs}) == 352
    dev = {(e.family, e.question.split()[0]) for e in exs if e.split == "dev"}
    assert sum(e.split == "dev" for e in exs) == sum(e.split == "test" for e in exs) == 176
    # every family is in both splits, but a given wording is in only one
    assert {e.family for e in exs if e.split == "dev"} == {e.family for e in exs if e.split == "test"}
    dev_q = {e.question for e in exs if e.split == "dev"}
    assert not dev_q & {e.question for e in exs if e.split == "test"}
    assert dev


def test_every_gold_query_runs_and_passes_the_critic_with_its_contract(con):
    critic = Critic()
    for e in golden.generate():
        res = run_sql(con, e.gold_sql)
        assert res.ok, (e.id, res.error)
        assert critic.review(e.gold_sql, res, e.contract).ok, e.id


def test_contract_inference_is_exact_on_dev_wording():
    for e in golden.load("dev"):
        c = infer_contract(e.question)
        assert (c.shape, c.k) == (e.contract.shape, e.contract.k), e.question


def test_fk_descendants():
    assert descendants("orders") == {"order_items"}
    assert descendants("regions") == {"customers", "orders", "order_items"}
    assert descendants("order_items") == set()
