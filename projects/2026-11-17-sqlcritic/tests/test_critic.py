import pytest

from sqlcritic.contract import Contract
from sqlcritic.critic import Critic
from sqlcritic.execute import run_sql

critic = Critic()


def codes(con, sql, contract):
    res = run_sql(con, sql) if ";" not in sql else None
    return critic.review(sql, res, contract).codes()


SCALAR = Contract("scalar")


@pytest.mark.parametrize("sql,contract,expected", [
    ("SELEC COUNT(*) FROM orders", SCALAR, "parse_error"),
    ("DELETE FROM orders", SCALAR, "not_read_only"),
    ("SELECT 1; DROP TABLE orders", SCALAR, "not_read_only"),
    ("SELECT COUNT(*) FROM invoices", SCALAR, "unknown_table"),
    ("SELECT SUM(o.amount) FROM orders o", SCALAR, "unknown_column"),
    ("SELECT COUNT(*) FROM orders o JOIN order_items oi ON oi.order_id = o.order_id",
     Contract("scalar", count_entity="orders"), "fanout_count"),
    ("SELECT COUNT(*) FROM orders o JOIN order_items oi ON oi.order_id = o.order_id",
     SCALAR, "fanout_count"),                               # counted table = FROM table
    ("SELECT AVG(p.unit_price) FROM products p JOIN order_items oi ON oi.product_id = p.product_id",
     SCALAR, "fanout_aggregate"),
    ("SELECT status, COUNT(*) FROM orders GROUP BY status", SCALAR, "column_count"),
    ("SELECT COUNT(*) FROM orders GROUP BY status", SCALAR, "row_count"),
    ("SELECT SUM(oi.quantity) FROM order_items oi WHERE oi.quantity > 99", SCALAR, "null_result"),
    ("SELECT MAX(name) FROM regions", SCALAR, "measure_not_numeric"),
    ("SELECT name, region_id FROM regions WHERE region_id > 99", Contract("grouped"), "empty_result"),
    ("SELECT r.name, COUNT(*) FROM customers c JOIN regions r ON r.region_id = c.region_id "
     "GROUP BY c.segment, r.name", Contract("grouped"), "duplicate_group_keys"),
    ("SELECT p.name, p.unit_price FROM products p", Contract("topk", k=5), "row_count"),
    ("SELECT p.name, p.unit_price FROM products p LIMIT 5", Contract("topk", k=5), "missing_order_limit"),
    ("SELECT p.name, p.unit_price FROM products p ORDER BY p.unit_price ASC LIMIT 5",
     Contract("topk", k=5), "wrong_order"),
    ("SELECT strftime('%m', order_date) AS m, COUNT(*) FROM orders GROUP BY m ORDER BY m DESC",
     Contract("series"), "wrong_order"),
    ("SELECT COUNT(*) FROM orders o, orders o2, orders o3", SCALAR, "exec_error"),  # budget
])
def test_issue_codes(con, sql, contract, expected):
    assert expected in codes(con, sql, contract)


@pytest.mark.parametrize("sql,contract", [
    ("SELECT COUNT(DISTINCT o.order_id) FROM orders o JOIN order_items oi ON oi.order_id = o.order_id",
     Contract("scalar", count_entity="orders")),
    ("SELECT COUNT(*) FROM customers c JOIN regions r ON r.region_id = c.region_id WHERE r.name = 'North'",
     Contract("scalar", count_entity="customers")),            # parent join does not fan out
    ("WITH t AS (SELECT o.status AS s FROM orders o) SELECT s, COUNT(*) FROM t GROUP BY s",
     Contract("grouped")),
    ("SELECT p.name, p.unit_price FROM products p WHERE p.unit_price > 99999", Contract("list")),
])
def test_valid_queries_pass(con, sql, contract):
    assert codes(con, sql, contract) == []


def test_feedback_message_names_the_contract(con):
    v = critic.review("SELECT COUNT(*) FROM orders", run_sql(con, "SELECT COUNT(*) FROM orders"),
                      Contract("grouped"))
    assert v.issues[0].code == "column_count"
    assert "one row per group" in v.issues[0].message
