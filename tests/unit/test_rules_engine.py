"""Unit tests for the rules engine and merchant mapping."""

from decimal import Decimal
import pytest

from core.classification.features import extract_features
from core.classification.rules_engine import RulesEngine


@pytest.fixture
def rules_engine():
    return RulesEngine()


def test_atm_cash_withdrawal_rule(rules_engine):
    feat = extract_features("ATM WDL/899127/MUMBAI", "DR", Decimal("5000.00"))
    res = rules_engine.classify(feat)
    assert res.matched is True
    assert res.category == "ATM/Cash Withdrawal"
    assert res.confidence >= 0.90


def test_salary_credit_rule(rules_engine):
    feat = extract_features("NEFT*HDFC0267640*SALARY CREDIT APRIL", "CR", Decimal("75000.00"))
    res = rules_engine.classify(feat)
    assert res.matched is True
    assert res.category == "Salary/Income"


def test_merchant_swiggy_food_dining(rules_engine):
    feat = extract_features("UPI/608946209524/Payment to Swiggy/swiggy@icici", "DR", Decimal("450.00"))
    res = rules_engine.classify(feat)
    assert res.matched is True
    assert res.category == "Food & Dining"


def test_merchant_amazon_shopping(rules_engine):
    feat = extract_features("POS 5234XXXXXX3194 AMAZON PAY DELHI", "DR", Decimal("1499.00"))
    res = rules_engine.classify(feat)
    assert res.matched is True
    assert res.category in ("Shopping/E-commerce", "Utilities")


def test_irctc_transport(rules_engine):
    feat = extract_features("IRCTC TRAIN TICKETING NEW DELHI", "DR", Decimal("850.00"))
    res = rules_engine.classify(feat)
    assert res.matched is True
    assert res.category == "Transport/Fuel/Travel"

