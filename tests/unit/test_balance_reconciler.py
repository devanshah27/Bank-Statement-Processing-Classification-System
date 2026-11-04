"""Unit tests for balance reconciler and anomaly detection."""

from decimal import Decimal
import pytest

from core.validation.balance_reconciler import reconcile, ReconciliationReport


def test_perfect_reconciliation():
    rows = [
        {"row_id": "1", "date": "2026-04-01", "description": "Deposit", "debit": None, "credit": 1000.0, "balance": 11000.0},
        {"row_id": "2", "date": "2026-04-02", "description": "Groceries", "debit": 500.0, "credit": None, "balance": 10500.0},
        {"row_id": "3", "date": "2026-04-03", "description": "Dining", "debit": 200.0, "credit": None, "balance": 10300.0},
    ]
    rep: ReconciliationReport = reconcile(
        rows,
        opening_balance=Decimal("10000.00"),
        closing_balance=Decimal("10300.00"),
    )
    assert rep.reconciliation_rate == 1.0
    assert rep.summary_balance_ok is True
    assert rep.auto_corrected_rows == 0
    assert len(rep.duplicate_indices) == 0


def test_swapped_debit_credit_auto_correction():
    # Row 2 shows credit=500 instead of debit=500, but balance went DOWN by 500!
    rows = [
        {"row_id": "1", "date": "2026-04-01", "description": "Deposit", "debit": None, "credit": 1000.0, "balance": 11000.0},
        {"row_id": "2", "date": "2026-04-02", "description": "Misprinted Debit as Credit", "debit": None, "credit": 500.0, "balance": 10500.0},
    ]
    rep = reconcile(
        rows,
        opening_balance=Decimal("10000.00"),
        closing_balance=Decimal("10500.00"),
    )
    assert rep.auto_corrected_rows == 1
    assert rep.row_results[1].was_auto_corrected is True
    assert "swapped" in rep.row_results[1].correction_applied.lower()


def test_duplicate_row_detection():
    rows = [
        {"row_id": "1", "date": "2026-04-01", "description": "Coffee", "debit": 150.0, "credit": None, "balance": 9850.0},
        {"row_id": "2", "date": "2026-04-01", "description": "Coffee", "debit": 150.0, "credit": None, "balance": 9850.0},
    ]
    rep = reconcile(rows, opening_balance=Decimal("10000.00"), closing_balance=Decimal("9850.00"))
    assert len(rep.duplicate_indices) >= 1


def test_zero_amount_detection():
    rows = [
        {"row_id": "1", "date": "2026-04-01", "description": "Zero txn", "debit": 0.0, "credit": 0.0, "balance": 10000.0},
    ]
    rep = reconcile(rows, opening_balance=Decimal("10000.00"), closing_balance=Decimal("10000.00"))
    assert 0 in rep.zero_amount_indices


def test_reversal_pair_detection():
    rows = [
        {"row_id": "1", "date": "2026-04-01", "description": "ATM WDL CHARGES", "debit": 23.60, "credit": None, "balance": 9976.40},
        {"row_id": "2", "date": "2026-04-02", "description": "REVERSAL OF CHG", "debit": None, "credit": 23.60, "balance": 10000.00},
    ]
    rep = reconcile(rows, opening_balance=Decimal("10000.00"), closing_balance=Decimal("10000.00"))
    assert len(rep.reversal_pairs) >= 1

