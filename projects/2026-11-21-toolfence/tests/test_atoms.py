from toolfence.atoms import atoms_of_value, extract


def test_email_case_and_trailing_dot():
    assert extract("email", "Write to Alice.Chen@Acme-Corp.com.") == {"alice.chen@acme-corp.com"}


def test_accounts_internal_and_spaced_iban_canonicalise():
    text = "pay ACCT-3301-7720 or DE89 3704 0044 0532 0130 00"
    assert extract("account", text) == {"ACCT-3301-7720", "DE89370400440532013000"}
    assert atoms_of_value("account", "de89 3704 0044 0532 0130 00".upper()) == {"DE89370400440532013000"}


def test_amounts_need_currency_marking():
    assert extract("amount", "Rent is $1,450.00 or 1450 EUR") == {"1450.00"}
    # dates, ids and bare numbers do not count as money in prose
    assert extract("amount", "On 2026-11-21 ticket 5512 was 1450") == set()
    assert extract("amount", "€3,200.00") == {"3200.00"}


def test_amount_argument_forms():
    assert atoms_of_value("amount", 1450) == {"1450.00"}
    assert atoms_of_value("amount", "1,450.5") == {"1450.50"}
    assert atoms_of_value("amount", "$60") == {"60.00"}
    assert atoms_of_value("amount", True) == set()


def test_hosts_from_urls_and_domains_but_not_emails():
    text = "see https://www.News.example.org/x?y=1 and blog.example.com, mail bob@corp.com"
    assert extract("host", text) == {"news.example.org", "blog.example.com"}


def test_paths_exclude_urls_and_emails():
    text = "delete reports/q3-summary.md (not https://x.example/a/b.txt nor me@a.example)"
    assert extract("path", text) == {"reports/q3-summary.md"}
    assert atoms_of_value("path", "./reports/../reports/Q3-summary.md") == {"reports/q3-summary.md"}


def test_unparseable_value_has_no_atoms():
    assert atoms_of_value("email", "grab at leaky docs") == set()
    assert atoms_of_value("account", "the usual one") == set()
