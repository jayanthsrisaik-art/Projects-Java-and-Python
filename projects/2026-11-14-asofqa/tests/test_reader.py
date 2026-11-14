from types import SimpleNamespace

from asofqa.reader import ClaudeReader, ExtractiveReader
from tests.helpers import MEAL, PTO, mini_corpus


def test_reads_value_not_constraint_or_history():
    c = mini_corpus()
    r = ExtractiveReader()
    # pto-v3 lists a procedural "10 days" first and mentions the replaced 20 days later
    assert r.read_value("How many days of paid time off do employees get?", PTO[2], c) == "25"
    assert r.read_value("What is the daily meal allowance?", MEAL[1], c) == "65"


def test_history_filter_matters_when_history_is_the_best_lexical_match():
    c = mini_corpus()
    q = "Which figure did this version replace?"
    assert ExtractiveReader(history_filter=True).read_value(q, PTO[3], c) == "27"
    assert ExtractiveReader(history_filter=False).read_value(q, PTO[3], c) == "25"


class FakeClient:
    def __init__(self, text, stop_reason="end_turn"):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self._resp = SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])

    def _create(self, **kw):
        self.calls.append(kw)
        return self._resp


def test_claude_reader_parses_number_and_sends_document():
    c = mini_corpus()
    fake = FakeClient("$2,000")
    assert ClaudeReader(client=fake).read_value("budget?", PTO[0], c) == "2000"
    assert "20 days of paid time off" in fake.calls[0]["messages"][0]["content"]
    assert fake.calls[0]["model"] == "claude-opus-5-5"


def test_claude_reader_none_and_refusal():
    c = mini_corpus()
    assert ClaudeReader(client=FakeClient("NONE")).read_value("q", PTO[0], c) is None
    assert ClaudeReader(client=FakeClient("25", stop_reason="refusal")).read_value("q", PTO[0], c) is None
