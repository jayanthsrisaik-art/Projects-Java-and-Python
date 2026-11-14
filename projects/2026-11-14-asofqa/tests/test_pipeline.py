from datetime import date

import pytest

from asofqa.pipeline import Answerer
from tests.helpers import mini_corpus

REF = date(2026, 11, 14)
Q = "paid time off days"


@pytest.fixture(scope="module")
def corpus():
    return mini_corpus()


def ask(corpus, q, strategy="asof"):
    return Answerer(corpus, REF, strategy).answer(q)


def test_current_ignores_scheduled_and_proposed(corpus):
    a = ask(corpus, f"How many {Q} do employees get?")
    assert a.values == ["25"] and a.evidence[0].doc_id == "pto-v3"
    assert "pto-v3" in a.text


def test_point_in_time_uses_version_in_force_and_skips_withdrawn(corpus):
    # pto-v2 (30 days) was planned for 2022-03-01 but withdrawn
    assert ask(corpus, f"How many {Q} on 2022-06-01?").values == ["20"]
    assert ask(corpus, f"How many {Q} on 2022-06-01?", "asof_nostatus").values == ["30"]


def test_range_reports_every_value_in_force(corpus):
    a = ask(corpus, f"How many {Q} in 2023?")
    assert a.values == ["20", "25"]
    assert [e.valid_to for e in a.evidence] == ["2023-06-30", "2026-12-31"]
    assert "changed during that period" in a.text


def test_future_point_counts_approved_schedule_not_draft(corpus):
    assert ask(corpus, f"How many {Q} on 2027-08-01?").values == ["27"]


def test_before_first_version_abstains(corpus):
    a = ask(corpus, f"How many {Q} in 2019?")
    assert a.abstain and a.values == []


def test_previous(corpus):
    assert ask(corpus, f"What were the {Q} before the most recent change?").values == ["20"]


def test_when(corpus):
    a = ask(corpus, f"When did the {Q} become 25 days?")
    assert a.date == "2023-07-01"
    assert ask(corpus, f"When will the {Q} become 27 days?").date == "2027-01-01"
    # withdrawn and draft values never took effect
    assert ask(corpus, f"When did the {Q} become 30 days?").abstain
    assert ask(corpus, f"When did the {Q} become 35 days?").abstain


def test_latest_baseline_is_right_now_but_wrong_for_history(corpus):
    assert ask(corpus, f"How many {Q} do employees get?", "latest").values == ["25"]
    assert ask(corpus, f"How many {Q} on 2022-06-01?", "latest").values == ["25"]


def test_policy_routing(corpus):
    a = ask(corpus, "What was the daily meal allowance for business travel in March 2022?")
    assert (a.policy, a.values) == ("meal", ["50"])


def test_unknown_strategy(corpus):
    with pytest.raises(ValueError):
        Answerer(corpus, REF, "magic")
