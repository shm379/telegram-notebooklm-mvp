from telegram_notebook.bot import NotebookBot, normalize_code, normalize_phone


def test_normalize_phone_basic_and_persian():
    assert normalize_phone("09123456789") == "+09123456789"
    assert normalize_phone("۰۹۱۲۳۴۵۶۷۸۹") == "+09123456789"
    assert normalize_phone("+98 912 345 6789") == "+989123456789"


def test_normalize_phone_rejects_out_of_range():
    assert normalize_phone("12345") is None
    assert normalize_phone("1234567890123456") is None


def test_normalize_code_variants():
    assert normalize_code("12345") == "12345"
    assert normalize_code("1-2-3 4-5") == "12345"
    assert normalize_code("۱۲۳۴۵") == "12345"


def test_normalize_code_bounds():
    assert normalize_code("12") is None
    assert normalize_code("1234567890123") is None
    assert normalize_code("") is None


def test_split_source():
    assert NotebookBot._split_source("q --source https://t.me/x") == ("q", "https://t.me/x")
    assert NotebookBot._split_source("  just a query ") == ("just a query", None)
    assert NotebookBot._split_source("") == ("", None)
