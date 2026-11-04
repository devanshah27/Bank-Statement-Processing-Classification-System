"""Unit tests for normalizers."""

from datetime import date
from decimal import Decimal
import pytest

from core.extraction.normalizers import (
    normalize_date,
    normalize_amount,
    extract_amount_and_direction,
    clean_ocr_text,
    clean_numeric_ocr,
    normalize_description,
    extract_payment_channel,
    format_indian_number,
    is_zero_amount,
)


def test_normalize_date_formats():
    assert normalize_date("01/04/2026") == date(2026, 4, 1)
    assert normalize_date("15-05-26", year_hint=2026) == date(2026, 5, 15)
    assert normalize_date("2026-06-30") == date(2026, 6, 30)
    assert normalize_date("01 Jan 2026") == date(2026, 1, 1)
    assert normalize_date("Apr 12, 2026") == date(2026, 4, 12)
    assert normalize_date("01/04/2026 01/04/2026") == date(2026, 4, 1)


def test_normalize_date_invalid():
    assert normalize_date("") is None
    assert normalize_date("INVALID_DATE_STRING") is None


def test_normalize_amount_indian():
    val = normalize_amount("1,23,456.78")
    assert val == Decimal("123456.78")


def test_normalize_amount_western():
    val = normalize_amount("1,234,567.89")
    assert val == Decimal("1234567.89")


def test_normalize_amount_bracketed():
    val = normalize_amount("(1,250.00)")
    assert val == Decimal("-1250.00")


def test_normalize_amount_dr_cr_suffix():
    val, d = extract_amount_and_direction("1,250.00 Dr")
    assert val == Decimal("1250.00")
    assert d == "DR"

    val_cr, d_cr = extract_amount_and_direction("5,000.00 Cr")
    assert val_cr == Decimal("5000.00")
    assert d_cr == "CR"


def test_normalize_amount_signed():
    val = normalize_amount("-1,250.00")
    assert val == Decimal("-1250.00")


def test_normalize_amount_zero_and_nil():
    assert is_zero_amount(normalize_amount("0.00"))
    assert is_zero_amount(normalize_amount("-"))
    assert is_zero_amount(normalize_amount("NIL"))


def test_clean_ocr_text():
    cleaned = clean_numeric_ocr("1O,OOO.5O")
    assert "10,000.50" in cleaned


def test_extract_payment_channel():
    assert extract_payment_channel("UPI/DR/123456/SWIGGY") == "UPI"
    assert extract_payment_channel("NEFT*HDFC*123*COMPANY*SALARY") == "NEFT"
    assert extract_payment_channel("IMPS/456/RENT") == "IMPS"
    assert extract_payment_channel("ATM WDL/MUMBAI") == "ATM"
    assert extract_payment_channel("POS STARBUCKS COFFEE") == "POS"


def test_format_indian_number():
    assert format_indian_number(Decimal("123456.78")) == "1,23,456.78"
    assert format_indian_number(Decimal("1000.00")) == "1,000.00"
    assert format_indian_number(Decimal("50.50")) == "50.50"
