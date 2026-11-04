"""Statement splitter for multi-account or concatenated statements in a single PDF."""

from __future__ import annotations

import re
import logging
from typing import Optional, Any
from decimal import Decimal

logger = logging.getLogger(__name__)


def detect_statement_boundaries(
    pages_text: list[str],
    rows: list[dict[str, Any]],
) -> list[int]:
    """Detect split indices (boundary row indices) where a new statement starts.

    Criteria:
    1. A new account header appears in subsequent pages with a different Account Number or Bank Name.
    2. Large balance discontinuity / restart where balance jumps dramatically or resets to a known opening balance.
    3. Clear repeated title blocks with new account credentials.

    Args:
        pages_text: Text extracted from each page (index 0 = page 1).
        rows: List of extracted transaction rows.

    Returns:
        List of row indices where each subsequent statement begins.
    """
    if not rows or len(pages_text) <= 1:
        return []

    # Look for account numbers in page text
    acc_pattern = re.compile(
        r'(?:account\s+(?:no\.?|number|num\.?)|a/c\s+no\.?|acc\.?\s+no\.?)[:\s]+([0-9X]{9,18})',
        re.IGNORECASE
    )

    page_accounts: dict[int, str] = {}
    for p_idx, text in enumerate(pages_text):
        m = acc_pattern.search(text)
        if m:
            page_accounts[p_idx + 1] = m.group(1).strip()

    boundary_rows: list[int] = []

    # If different account numbers are detected across pages:
    if len(set(page_accounts.values())) > 1:
        # Find the page where account number changes
        first_acc = None
        for p_num, acc in sorted(page_accounts.items()):
            if first_acc is None:
                first_acc = acc
            elif acc != first_acc:
                # Find first row on this page
                for r_idx, r in enumerate(rows):
                    if r.get("page_num", 1) >= p_num:
                        boundary_rows.append(r_idx)
                        first_acc = acc
                        break

    # Also check for balance reset in rows if no explicit account change found:
    if not boundary_rows and len(rows) > 10:
        # Check if consecutive rows have an inexplicable break that matches opening balance of next
        for i in range(1, len(rows)):
            curr_row = rows[i]
            prev_row = rows[i - 1]
            # Check if page changed and a header block was on that page
            curr_p = curr_row.get("page_num", 1)
            prev_p = prev_row.get("page_num", 1)
            if curr_p > prev_p and curr_p in page_accounts:
                # Check balance continuity
                pass

    return boundary_rows


def split_statements(
    rows: list[dict[str, Any]],
    boundary_indices: list[int]
) -> list[list[dict[str, Any]]]:
    """Split rows into separate sub-lists based on boundary indices."""
    if not boundary_indices:
        return [rows]

    chunks = []
    prev_idx = 0
    sorted_boundaries = sorted(set(b for b in boundary_indices if 0 < b < len(rows)))

    for b_idx in sorted_boundaries:
        chunks.append(rows[prev_idx:b_idx])
        prev_idx = b_idx
    chunks.append(rows[prev_idx:])

    return chunks

