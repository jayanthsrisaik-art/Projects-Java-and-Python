from types import SimpleNamespace

import pytest

from sqlcritic.graph import Pipeline
from sqlcritic.writers import (Attempt, ClaudeWriter, HeuristicWriter, WriterError, build_user_message,
                               extract_sql, parse_slots)

GOOD = "SELECT COUNT(*) FROM orders o WHERE strftime('%Y', o.order_date) = '2024'"
FANOUT = ("SELECT COUNT(*) FROM orders o JOIN order_items oi ON oi.order_id = o.order_id "
          "WHERE strftime('%Y', o.order_date) = '2024'")
BROKEN = "SELECT COUNT(*) FROM orderz"
Q = "How many orders were placed in 2024?"


class Scripted:
    name = "scripted"

    def __init__(self, *sqls):
        self.sqls = list(sqls)
        self.seen = []

    def write(self, question, history):
        self.seen.append([a.issues for a in history])
        if len(history) >= len(self.sqls):
            raise WriterError("out of ideas")
        return self.sqls[len(history)]


def test_critic_mode_rejects_fanout_and_feeds_back(con):
    w = Scripted(FANOUT, GOOD)
    ans = Pipeline(w, con, "critic").ask(Q)
    assert ans.status == "answered" and ans.sql == GOOD
    assert [i["code"] for i in ans.attempts[0].issues] == ["fanout_count"]
    assert w.seen[1][0][0]["code"] == "fanout_count"      # writer saw the critic's feedback


def test_single_mode_shows_the_wrong_answer(con):
    ans = Pipeline(Scripted(FANOUT, GOOD), con, "single").ask(Q)
    assert ans.status == "answered" and ans.sql == FANOUT


def test_exec_retry_only_retries_errors(con):
    assert Pipeline(Scripted(BROKEN, GOOD), con, "exec_retry").ask(Q).sql == GOOD
    assert Pipeline(Scripted(FANOUT, GOOD), con, "exec_retry").ask(Q).sql == FANOUT


def test_abstains_when_budget_is_spent(con):
    ans = Pipeline(Scripted(FANOUT, FANOUT, FANOUT), con, "critic", max_attempts=3).ask(Q)
    assert ans.status == "abstained" and ans.sql is None and ans.rows == []
    assert len(ans.attempts) == 3


def test_writer_error_abstains(con):
    ans = Pipeline(Scripted(), con, "critic").ask(Q)
    assert ans.status == "abstained"
    assert ans.attempts[0].issues[0]["code"] == "writer_error"


def test_heuristic_writer_switches_shape_after_shape_feedback(con):
    q = "Break down 2024 sales across regions."
    ans = Pipeline(HeuristicWriter(), con, "critic").ask(q)
    assert ans.attempts[0].issues[0]["code"] == "column_count"
    assert len(ans.columns) == 2 and len({r[0] for r in ans.rows}) == len(ans.rows)


def test_slot_parsing():
    s = parse_slots("Top 5 products by revenue for 2024 in the North region, over $50 smb")
    assert (s.k, s.year, s.region, s.price, s.segment) == (5, "2024", "North", "50", "smb")


# ------------------------------------------------------------- Claude writer with a fake client

class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        stop, text = self.replies.pop(0)
        return SimpleNamespace(stop_reason=stop, content=[
            SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)])


def fake_client(*replies):
    msgs = FakeMessages(replies)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


def test_claude_writer_request_and_feedback_loop(con):
    client, msgs = fake_client(("end_turn", f"```sql\n{FANOUT};\n```"),
                               ("end_turn", f"Fixed:\n```sql\n{GOOD}\n```"))
    ans = Pipeline(ClaudeWriter(client=client), con, "critic").ask(Q)
    assert ans.sql == GOOD
    first, second = msgs.calls
    assert first["model"] == "claude-opus-5-5"
    assert first["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in first["betas"]
    assert "revenue" in first["system"] and "order_items" in first["system"]
    assert "fanout_count" in second["messages"][0]["content"]
    assert "Attempt 1" in second["messages"][0]["content"]


def test_claude_writer_refusal_and_truncation():
    for stop in ("refusal", "max_tokens"):
        client, _ = fake_client((stop, ""))
        with pytest.raises(WriterError):
            ClaudeWriter(client=client).write(Q, [])


def test_extract_sql_and_prompt():
    assert extract_sql("```SQL\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("SELECT 2") == "SELECT 2"
    with pytest.raises(WriterError):
        extract_sql("```sql\n```")
    msg = build_user_message(Q, [Attempt("SELECT 1", [{"code": "row_count", "message": "m"}])])
    assert "[row_count] m" in msg and msg.endswith("Write a corrected query.")
