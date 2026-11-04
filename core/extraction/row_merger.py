"""Merge multi-line narration rows into single transaction records.

Bank statement parsers often produce raw rows where a single transaction's
description spans multiple physical lines. This module identifies continuation
rows and folds them back into the primary transaction row.

A **continuation row** is defined as a row where ALL of the following fields
are empty: date, debit, credit, balance.

Additionally, the merger strips:

- ``BROUGHT FORWARD`` / ``BALANCE B/F`` opening balance rows
- ``CARRIED FORWARD`` / ``BAL C/F`` closing balance rows
- Repeated table header rows
- Page sub-total rows (``TOTAL``, ``SUB TOTAL``, ``PAGE TOTAL``)
- Disclaimer / footer noise rows
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & patterns
# ---------------------------------------------------------------------------

#: Row fields that must all be absent for a row to be a pure continuation.
_AMOUNT_FIELDS: tuple[str, ...] = (
    "date", "date_raw", "debit", "debit_raw", "credit", "credit_raw",
    "balance", "balance_raw", "amount", "amount_raw"
)

#: Keywords that identify opening/closing carry-forward sentinel rows.
_CARRY_FORWARD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bBROUGHT\s+FORWARD\b", re.IGNORECASE),
    re.compile(r"\bBALANCE\s+B/?F\b", re.IGNORECASE),
    re.compile(r"\bOPENING\s+BAL(?:ANCE)?\b", re.IGNORECASE),
    re.compile(r"\bCARRIED\s+FORWARD\b", re.IGNORECASE),
    re.compile(r"\bBAL(?:ANCE)?\s+C/?F\b", re.IGNORECASE),
    re.compile(r"\bCLOSING\s+BAL(?:ANCE)?\b", re.IGNORECASE),
]

#: Keywords that identify repeated header rows (column titles).
_HEADER_KEYWORDS: frozenset[str] = frozenset({
    "date", "narration", "description", "particulars", "chq", "cheque",
    "ref", "withdrawal", "deposit", "debit", "credit", "balance",
    "value date", "txn date", "transaction",
})

#: Minimum number of header keyword matches to classify a row as a header.
_HEADER_MATCH_THRESHOLD = 3

#: Keywords that identify page total / sub-total rows.
_TOTAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bPAGE\s+TOTAL\b", re.IGNORECASE),
    re.compile(r"\bSUB[\s\-]?TOTAL\b", re.IGNORECASE),
    re.compile(r"^\s*TOTAL\s*$", re.IGNORECASE),
    re.compile(r"\bGRAND\s+TOTAL\b", re.IGNORECASE),
]

#: Patterns for disclaimer / footer noise rows.
_FOOTER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"This\s+is\s+a\s+computer[\s\-]generated", re.IGNORECASE),
    re.compile(r"This\s+statement\s+is\s+(system|computer)", re.IGNORECASE),
    re.compile(r"E\s*&\s*O\.?\s*E\.?", re.IGNORECASE),
    re.compile(r"Errors?\s+and\s+Omissions\s+Excepted", re.IGNORECASE),
    re.compile(r"For\s+queries\s+(contact|call|write)", re.IGNORECASE),
    re.compile(r"www\.[a-z]+\.com", re.IGNORECASE),
    re.compile(r"CIN\s*[:\-]?\s*[A-Z]\d+", re.IGNORECASE),
    re.compile(r"Registered\s+Office", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_description(row: dict[str, Any]) -> str:
    """Return the description / narration text from a row dict."""
    for key in ("description", "narration", "particulars", "details", "remarks"):
        val = row.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return ""


def _set_description(row: dict[str, Any], text: str) -> None:
    """Write *text* back to whichever description key the row uses."""
    for key in ("description", "narration", "particulars", "details", "remarks"):
        if key in row:
            row[key] = text
            return
    # Fallback: create a 'description' key
    row["description"] = text


def _is_continuation(row: dict[str, Any]) -> bool:
    """Return ``True`` if *row* is a pure narration continuation line.

    A continuation row has empty (or missing) date, debit, credit, and balance.
    """
    for field in _AMOUNT_FIELDS:
        val = row.get(field, "")
        if val and str(val).strip():
            return False
    return True


def _is_carry_forward(row: dict[str, Any]) -> bool:
    """Return ``True`` if *row* is an opening/closing carry-forward sentinel."""
    description = _row_description(row)
    for pattern in _CARRY_FORWARD_PATTERNS:
        if pattern.search(description):
            return True
    return False


def _is_header_row(row: dict[str, Any]) -> bool:
    """Return ``True`` if *row* looks like a repeated column header line."""
    description = _row_description(row).lower()
    # Also check the date and balance fields for column-header text
    all_text = " ".join(str(v) for v in row.values() if v).lower()

    matches = sum(1 for kw in _HEADER_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", all_text))
    return matches >= _HEADER_MATCH_THRESHOLD


def _is_total_row(row: dict[str, Any]) -> bool:
    """Return ``True`` if *row* is a page or sub-total row."""
    description = _row_description(row)
    for pattern in _TOTAL_PATTERNS:
        if pattern.search(description):
            return True
    return False


def _is_footer_row(row: dict[str, Any]) -> bool:
    """Return ``True`` if *row* is a disclaimer or footer line."""
    description = _row_description(row)
    for pattern in _FOOTER_PATTERNS:
        if pattern.search(description):
            return True
    return False


def _has_balance_on_continuation(row: dict[str, Any]) -> bool:
    """Return ``True`` if the row has only a balance value (no date/debit/credit)."""
    has_balance = bool(row.get("balance", ""))
    has_date = bool(str(row.get("date", "")).strip())
    has_debit = bool(str(row.get("debit", "")).strip())
    has_credit = bool(str(row.get("credit", "")).strip())
    return has_balance and not has_date and not has_debit and not has_credit


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_multiline_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge continuation narration rows into their parent transaction row.

    Parameters
    ----------
    raw_rows:
        List of raw row dicts as produced by a bank-profile table parser.
        Each dict should contain at minimum the keys ``date``, ``debit``,
        ``credit``, and ``balance`` (or ``amount``), plus a description key
        (``description``, ``narration``, or ``particulars``).

    Returns
    -------
    list[dict]
        Cleaned, merged list of transaction rows. The number of rows will be
        ≤ ``len(raw_rows)``.

    Notes
    -----
    - Carry-forward, header, total, and footer rows are stripped before the
      merge so that they do not accidentally absorb continuation lines.
    - If the first row of the input (after stripping) is a continuation row
      (no date), it is kept as a standalone row to avoid data loss.
    - Balance values found on continuation rows are promoted to the parent
      transaction if the parent has no balance yet.
    """
    if not raw_rows:
        return []

    merged: list[dict[str, Any]] = []
    rows_seen = 0
    rows_stripped = 0

    for row in raw_rows:
        rows_seen += 1

        # --- Strip noise rows ---
        if _is_carry_forward(row):
            logger.debug("Stripped carry-forward row: %.80s", _row_description(row))
            rows_stripped += 1
            continue

        if _is_header_row(row):
            logger.debug("Stripped header row: %.80s", _row_description(row))
            rows_stripped += 1
            continue

        if _is_total_row(row):
            logger.debug("Stripped total row: %.80s", _row_description(row))
            rows_stripped += 1
            continue

        if _is_footer_row(row):
            logger.debug("Stripped footer row: %.80s", _row_description(row))
            rows_stripped += 1
            continue

        # --- Merge continuation rows ---
        if _is_continuation(row) and merged:
            parent = merged[-1]
            parent_desc = _row_description(parent)
            cont_desc = _row_description(row)

            if cont_desc:
                merged_desc = f"{parent_desc} {cont_desc}".strip()
                _set_description(parent, merged_desc)
                logger.debug(
                    "Merged continuation: '%.40s' + '%.40s'", parent_desc, cont_desc
                )

            # Promote balance from continuation row if parent lacks one
            if _has_balance_on_continuation(row) and not parent.get("balance", ""):
                parent["balance"] = row["balance"]
                logger.debug(
                    "Promoted balance '%s' from continuation to parent", row["balance"]
                )

            rows_stripped += 1  # counts as merged / stripped
            continue

        # --- Normal transaction row ---
        merged.append(row)

    logger.info(
        "merge_multiline_rows: input=%d, output=%d, stripped/merged=%d",
        rows_seen,
        len(merged),
        rows_stripped,
    )
    return merged
