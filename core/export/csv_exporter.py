"""CSV export module encoded in UTF-8-sig for seamless spreadsheet compatibility."""

from __future__ import annotations

import csv
import io
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


def export_to_csv(
    transactions: list[dict[str, Any]],
    include_classification: bool = True
) -> bytes:
    """Export transactions to UTF-8-sig encoded CSV bytes.

    Args:
        transactions: List of transaction dictionaries.
        include_classification: Whether to include category and confidence columns.

    Returns:
        bytes representing the CSV file with UTF-8 BOM (utf-8-sig).
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    headers = ["date", "value_date", "description", "ref_no", "debit", "credit", "balance"]
    if include_classification:
        headers.extend(["category", "confidence", "source", "is_flagged", "flag_reasons"])

    writer.writerow(headers)

    for tx in transactions:
        row = [
            tx.get("date", ""),
            tx.get("value_date", ""),
            tx.get("description", ""),
            tx.get("ref_no", ""),
            tx.get("debit", "") if tx.get("debit") is not None else "",
            tx.get("credit", "") if tx.get("credit") is not None else "",
            tx.get("balance", "") if tx.get("balance") is not None else "",
        ]
        if include_classification:
            conf = tx.get("confidence")
            conf_str = f"{float(conf):.4f}" if conf is not None else ""
            flags = tx.get("flag_reasons", [])
            flags_str = "; ".join(flags) if isinstance(flags, list) else str(flags)
            is_fl = "true" if (tx.get("is_flagged") or flags) else "false"

            row.extend([
                tx.get("category", "Other/Uncategorized"),
                conf_str,
                tx.get("source", "rule"),
                is_fl,
                flags_str,
            ])
        writer.writerow(row)

    csv_str = output.getvalue()
    return csv_str.encode("utf-8-sig")

