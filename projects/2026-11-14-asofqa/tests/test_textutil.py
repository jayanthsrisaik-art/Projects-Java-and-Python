from asofqa.textutil import norm_number, quantities, sentences, tokens


def test_norm_number():
    assert norm_number("$2,000") == "2000"
    assert norm_number("0.670") == "0.67"
    assert norm_number("25") == "25"
    assert norm_number("1.50") == "1.5"


def test_quantities_require_a_unit():
    assert quantities("Employees accrue 6 paid sick days per year.") == ["6"]
    assert quantities("capped at 5% of salary, $0.655 per mile, $1,500 budget") == ["5", "0.655", "1500"]
    # dates, years and 401(k) are not quantities
    assert quantities("Effective 2024-03-01 under the 401(k) plan in 2023.") == []


def test_tokens_drop_stopwords_and_stem():
    assert tokens("What are the remote days?") == ["remote", "day"]


def test_sentences_keep_decimals_together():
    assert sentences("Rate is $0.67 per mile. Logs within 5 days.") == ["Rate is $0.67 per mile.", "Logs within 5 days."]
