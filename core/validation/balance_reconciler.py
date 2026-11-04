"""Balance chain validation and auto-correction.

This module validates the running balance chain of extracted transactions,
detecting and auto-correcting common extraction errors, flagging anomalies,
and producing a comprehensive reconciliation report.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Optional
from datetime import date as Date

logger = logging.getLogger(__name__)

# Maximum absolute difference still considered "equal" for balance comparison
BALANCE_TOLERANCE = Decimal("0.02")

# Maximum single-row balance jump to consider "unexplained" (heuristic)
_GAP_THRESHOLD_MULTIPLIER = Decimal("5")

# How many rows ahead to look for reversal pairs
_REVERSAL_WINDOW = 10

# Keywords that signal a reversal/clearing transaction
_REVERSAL_KEYWORDS: tuple[str, ...] = (
    "rev",
    "reversal",
    "clrg",
    "clearing",
    "refund",
    "chargeback",
)


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class ReconciliationResult:
    """Per-row outcome of the balance reconciliation check."""

    row_index: int
    expected_balance: Optional[Decimal]
    actual_balance: Optional[Decimal]
    difference: Decimal
    is_ok: bool
    was_auto_corrected: bool
    correction_applied: str   # e.g. "swapped debit/credit"
    flag_reason: str


@dataclass
class ReconciliationReport:
    """Aggregated report produced by :func:`reconcile`."""

    total_rows: int
    reconciled_rows: int
    failed_rows: int
    auto_corrected_rows: int
    reconciliation_rate: float          # 0.0 – 1.0
    summary_balance_ok: bool
    opening_balance: Optional[Decimal]
    closing_balance: Optional[Decimal]
    computed_closing: Optional[Decimal]
    row_results: list[ReconciliationResult]
    duplicate_indices: list[int]
    gap_indices: list[int]              # indices where balance jumps unexpectedly
    out_of_period_indices: list[int]
    zero_amount_indices: list[int]
    reversal_pairs: list[tuple[int, int]]
    warnings: list[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_decimal(value: object) -> Optional[Decimal]:
    """Safely coerce *value* to :class:`~decimal.Decimal`.

    Returns ``None`` when the value is absent or non-numeric.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        text = str(value).strip().replace(",", "")
        if text in ("", "-", "N/A", "n/a"):
            return None
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def _safe_zero(value: Optional[Decimal]) -> Decimal:
    """Return *value* or ``Decimal('0')`` if it is ``None``."""
    return value if value is not None else Decimal("0")


def _parse_date(value: object) -> Optional[Date]:
    """Try to extract a :class:`datetime.date` from *value*.

    Accepts ``datetime.date``, ``datetime.datetime``, or ISO-format strings.
    """
    if value is None:
        return None
    if isinstance(value, Date):
        return value
    # datetime.datetime inherits from datetime.date — already handled above
    if isinstance(value, str):
        from datetime import datetime
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d %b %Y", "%d-%b-%Y"):
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                continue
    return None


def _description_lower(row: dict) -> str:
    """Return lower-cased description from a transaction row dict."""
    for key in ("description", "narration", "particulars", "details"):
        val = row.get(key)
        if val:
            return str(val).lower()
    return ""


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def reconcile(
    rows: list[dict],
    opening_balance: Optional[Decimal],
    closing_balance: Optional[Decimal],
    tolerance: Decimal = BALANCE_TOLERANCE,
) -> ReconciliationReport:
    """Validate the balance chain in *rows* and produce a reconciliation report.

    Parameters
    ----------
    rows:
        List of transaction dicts.  Each dict is expected to have keys such as
        ``debit``, ``credit``, ``balance``, ``date``, and ``description``.
    opening_balance:
        Statement opening balance, or ``None`` if unknown.
    closing_balance:
        Statement closing balance, or ``None`` if unknown.
    tolerance:
        Maximum absolute difference in Decimal between expected and actual
        balance still considered a match.

    Returns
    -------
    ReconciliationReport
        Full reconciliation outcome including per-row results and all flags.
    """
    opening_balance = _to_decimal(opening_balance)
    closing_balance = _to_decimal(closing_balance)

    logger.info(
        "Starting reconciliation: %d rows, opening=%s, closing=%s",
        len(rows),
        opening_balance,
        closing_balance,
    )

    row_results: list[ReconciliationResult] = []
    warnings: list[str] = []
    duplicate_indices: list[int] = []
    gap_indices: list[int] = []
    out_of_period_indices: list[int] = []
    zero_amount_indices: list[int] = []
    reversal_pairs: list[tuple[int, int]] = []

    if not rows:
        logger.warning("reconcile() called with empty rows list")
        return ReconciliationReport(
            total_rows=0,
            reconciled_rows=0,
            failed_rows=0,
            auto_corrected_rows=0,
            reconciliation_rate=1.0,
            summary_balance_ok=(opening_balance == closing_balance),
            opening_balance=opening_balance,
            closing_balance=closing_balance,
            computed_closing=opening_balance,
            row_results=[],
            duplicate_indices=[],
            gap_indices=[],
            out_of_period_indices=[],
            zero_amount_indices=[],
            reversal_pairs=[],
            warnings=["No transaction rows provided"],
        )

    # ------------------------------------------------------------------
    # Pass 1 – per-row balance chain validation + auto-correction
    # ------------------------------------------------------------------
    total_credits = Decimal("0")
    total_debits = Decimal("0")

    # Determine statement date range for out-of-period detection
    dates: list[Date] = []
    for row in rows:
        d = _parse_date(row.get("date"))
        if d:
            dates.append(d)
    statement_min_date: Optional[Date] = min(dates) if dates else None
    statement_max_date: Optional[Date] = max(dates) if dates else None

    prev_balance: Optional[Decimal] = opening_balance
    reconciled = 0
    auto_corrected = 0

    for idx, row in enumerate(rows):
        debit = _to_decimal(row.get("debit"))
        credit = _to_decimal(row.get("credit"))
        actual_balance = _to_decimal(row.get("balance"))

        # --- Zero-amount detection ---
        if _safe_zero(debit) == Decimal("0") and _safe_zero(credit) == Decimal("0"):
            zero_amount_indices.append(idx)
            logger.debug("Row %d has zero debit and credit", idx)

        total_debits += _safe_zero(debit)
        total_credits += _safe_zero(credit)

        # --- Balance chain check ---
        expected_balance: Optional[Decimal] = None
        if prev_balance is not None:
            expected_balance = prev_balance - _safe_zero(debit) + _safe_zero(credit)

        is_ok = False
        was_auto_corrected = False
        correction_applied = ""
        flag_reason = ""
        difference = Decimal("0")

        if actual_balance is None or expected_balance is None:
            # Cannot verify — treat as ok but warn
            is_ok = True
            if actual_balance is None:
                flag_reason = "balance missing"
        else:
            difference = abs(actual_balance - expected_balance)
            if difference <= tolerance:
                is_ok = True
                reconciled += 1
            else:
                # --- Auto-correction attempt: swap debit/credit ---
                swapped_expected = prev_balance - _safe_zero(credit) + _safe_zero(debit)  # type: ignore[operator]
                swapped_diff = abs(actual_balance - swapped_expected)
                if swapped_diff <= tolerance:
                    # Swap fixes it → mutate the row and mark corrected
                    logger.info(
                        "Row %d: auto-correcting swapped debit/credit "
                        "(debit=%s, credit=%s)",
                        idx, debit, credit,
                    )
                    rows[idx]["debit"], rows[idx]["credit"] = (
                        rows[idx].get("credit"),
                        rows[idx].get("debit"),
                    )
                    # Update accumulators for the corrected values
                    total_debits -= _safe_zero(debit)
                    total_credits -= _safe_zero(credit)
                    total_debits += _safe_zero(credit)
                    total_credits += _safe_zero(debit)

                    expected_balance = swapped_expected
                    difference = swapped_diff
                    is_ok = True
                    was_auto_corrected = True
                    correction_applied = "swapped debit/credit"
                    flag_reason = "auto-corrected: swapped debit/credit"
                    reconciled += 1
                    auto_corrected += 1
                else:
                    # Check if balance delta proves debit or credit direction
                    bal_delta = actual_balance - prev_balance
                    row_amount = _safe_zero(debit) if _safe_zero(debit) > 0 else _safe_zero(credit)
                    # Check if debit and credit were split across decimal point (e.g. 2027.00 and 96.00)
                    comb1 = _safe_zero(credit) + (_safe_zero(debit) / Decimal("100"))
                    comb2 = _safe_zero(debit) + (_safe_zero(credit) / Decimal("100"))
                    is_split_match = abs(abs(bal_delta) - comb1) <= tolerance or abs(abs(bal_delta) - comb2) <= tolerance

                    if is_split_match:
                        correct_amount = comb1 if abs(abs(bal_delta) - comb1) <= tolerance else comb2
                        if bal_delta > 0:
                            logger.info("Row %d: recombining split decimal amount (+%s) as credit", idx, correct_amount)
                            rows[idx]["credit"] = float(correct_amount)
                            rows[idx]["debit"] = None
                            total_credits += correct_amount - _safe_zero(credit)
                            total_debits -= _safe_zero(debit)
                        else:
                            logger.info("Row %d: recombining split decimal amount (-%s) as debit", idx, correct_amount)
                            rows[idx]["debit"] = float(correct_amount)
                            rows[idx]["credit"] = None
                            total_debits += correct_amount - _safe_zero(debit)
                            total_credits -= _safe_zero(credit)
                        expected_balance = actual_balance
                        difference = Decimal("0")
                        is_ok = True
                        was_auto_corrected = True
                        correction_applied = "recombined split integer and cents amount"
                        flag_reason = "auto-corrected: recombined decimal amount"
                        reconciled += 1
                        auto_corrected += 1
                    elif row_amount > 0 and abs(abs(bal_delta) - row_amount) <= tolerance:
                        if bal_delta > 0:
                            # Balance increased -> MUST be credit
                            logger.info("Row %d: balance delta (+%s) proves credit inflow", idx, bal_delta)
                            rows[idx]["credit"] = float(bal_delta)
                            rows[idx]["debit"] = None
                            total_credits += abs(bal_delta)
                            total_debits -= _safe_zero(debit)
                        else:
                            # Balance decreased -> MUST be debit
                            logger.info("Row %d: balance delta (%s) proves debit outflow", idx, bal_delta)
                            rows[idx]["debit"] = float(abs(bal_delta))
                            rows[idx]["credit"] = None
                            total_debits += abs(bal_delta)
                            total_credits -= _safe_zero(credit)
                        expected_balance = actual_balance
                        difference = Decimal("0")
                        is_ok = True
                        was_auto_corrected = True
                        correction_applied = "balance-delta corrected direction"
                        flag_reason = "auto-corrected: balance delta proved direction"
                        reconciled += 1
                        auto_corrected += 1
                    else:
                        is_ok = False
                        flag_reason = (
                            f"balance mismatch: expected {expected_balance:.2f}, "
                            f"got {actual_balance:.2f} (diff {difference:.2f})"
                        )
                        logger.warning("Row %d reconciliation failure: %s", idx, flag_reason)

        # --- Gap detection: unexplained large jump ---
        if (
            prev_balance is not None
            and actual_balance is not None
            and expected_balance is not None
            and not is_ok
        ):
            # Already flagged above; additionally mark as gap if large
            jump = abs(actual_balance - prev_balance)
            avg_tx = (
                (total_debits + total_credits) / max(idx, 1)
                if idx > 0
                else Decimal("0")
            )
            if avg_tx > Decimal("0") and jump > avg_tx * _GAP_THRESHOLD_MULTIPLIER:
                gap_indices.append(idx)

        # Update prev_balance with the actual (or corrected expected) balance
        if actual_balance is not None:
            prev_balance = actual_balance
        elif expected_balance is not None:
            prev_balance = expected_balance

        row_results.append(
            ReconciliationResult(
                row_index=idx,
                expected_balance=expected_balance,
                actual_balance=actual_balance,
                difference=difference,
                is_ok=is_ok,
                was_auto_corrected=was_auto_corrected,
                correction_applied=correction_applied,
                flag_reason=flag_reason,
            )
        )

    # ------------------------------------------------------------------
    # Pass 2 – Duplicate detection
    # ------------------------------------------------------------------
    seen_signatures: dict[tuple, int] = {}
    for idx, row in enumerate(rows):
        debit = _to_decimal(row.get("debit"))
        credit = _to_decimal(row.get("credit"))
        # Bucket amounts to tolerance granularity
        d_bucket = int(_safe_zero(debit) / tolerance) if debit is not None else -1
        c_bucket = int(_safe_zero(credit) / tolerance) if credit is not None else -1
        desc = _description_lower(row)
        date_val = str(row.get("date", ""))
        sig = (date_val, desc[:60], d_bucket, c_bucket)
        if sig in seen_signatures:
            first = seen_signatures[sig]
            if first not in duplicate_indices:
                duplicate_indices.append(first)
            duplicate_indices.append(idx)
            logger.debug("Duplicate detected: row %d matches row %d", idx, first)
        else:
            seen_signatures[sig] = idx

    # ------------------------------------------------------------------
    # Pass 3 – Out-of-period detection
    # ------------------------------------------------------------------
    if statement_min_date and statement_max_date:
        for idx, row in enumerate(rows):
            d = _parse_date(row.get("date"))
            if d and (d < statement_min_date or d > statement_max_date):
                out_of_period_indices.append(idx)

    # ------------------------------------------------------------------
    # Pass 4 – Reversal pair detection
    # ------------------------------------------------------------------
    for idx, row in enumerate(rows):
        debit = _to_decimal(row.get("debit"))
        credit = _to_decimal(row.get("credit"))
        if debit is None or _safe_zero(debit) == Decimal("0"):
            continue  # only look for debits as anchors
        desc = _description_lower(row)
        for jdx in range(idx + 1, min(idx + 1 + _REVERSAL_WINDOW, len(rows))):
            future = rows[jdx]
            future_credit = _to_decimal(future.get("credit"))
            future_desc = _description_lower(future)
            if future_credit is None:
                continue
            if abs(debit - future_credit) <= tolerance and any(
                kw in future_desc for kw in _REVERSAL_KEYWORDS
            ):
                pair = (idx, jdx)
                if pair not in reversal_pairs:
                    reversal_pairs.append(pair)
                    logger.debug(
                        "Reversal pair detected: row %d debited %.2f, "
                        "row %d credited %.2f",
                        idx, debit, jdx, future_credit,
                    )
                break

    # ------------------------------------------------------------------
    # Summary balance check
    # ------------------------------------------------------------------
    computed_closing: Optional[Decimal] = None
    summary_balance_ok = False
    if opening_balance is not None:
        computed_closing = opening_balance + total_credits - total_debits
        if closing_balance is not None:
            diff = abs(computed_closing - closing_balance)
            summary_balance_ok = diff <= tolerance
            if not summary_balance_ok:
                msg = (
                    f"Summary balance mismatch: opening={opening_balance:.2f}, "
                    f"total_credits={total_credits:.2f}, "
                    f"total_debits={total_debits:.2f}, "
                    f"computed_closing={computed_closing:.2f}, "
                    f"stated_closing={closing_balance:.2f}, "
                    f"diff={diff:.2f}"
                )
                warnings.append(msg)
                logger.warning(msg)
        else:
            # No stated closing; flag as unknown
            warnings.append("Closing balance not provided; summary check skipped")
    else:
        warnings.append("Opening balance not provided; summary check skipped")

    # Additional warnings
    if duplicate_indices:
        warnings.append(f"{len(duplicate_indices)} potential duplicate row(s) detected")
    if gap_indices:
        warnings.append(f"{len(gap_indices)} unexplained balance gap(s) detected")
    if out_of_period_indices:
        warnings.append(
            f"{len(out_of_period_indices)} out-of-period transaction(s) detected"
        )
    if zero_amount_indices:
        warnings.append(
            f"{len(zero_amount_indices)} zero-amount transaction(s) detected"
        )
    if reversal_pairs:
        warnings.append(f"{len(reversal_pairs)} reversal pair(s) detected")

    failed = len(rows) - reconciled - auto_corrected + auto_corrected  # failed = not is_ok
    # Recount properly
    reconciled_count = sum(1 for r in row_results if r.is_ok)
    failed_count = len(rows) - reconciled_count
    rate = reconciled_count / len(rows) if rows else 1.0

    logger.info(
        "Reconciliation complete: %d/%d rows ok (%.1f%%), "
        "%d auto-corrected, summary_ok=%s",
        reconciled_count,
        len(rows),
        rate * 100,
        auto_corrected,
        summary_balance_ok,
    )

    return ReconciliationReport(
        total_rows=len(rows),
        reconciled_rows=reconciled_count,
        failed_rows=failed_count,
        auto_corrected_rows=auto_corrected,
        reconciliation_rate=rate,
        summary_balance_ok=summary_balance_ok,
        opening_balance=opening_balance,
        closing_balance=closing_balance,
        computed_closing=computed_closing,
        row_results=row_results,
        duplicate_indices=sorted(set(duplicate_indices)),
        gap_indices=sorted(set(gap_indices)),
        out_of_period_indices=sorted(set(out_of_period_indices)),
        zero_amount_indices=sorted(set(zero_amount_indices)),
        reversal_pairs=reversal_pairs,
        warnings=warnings,
    )
