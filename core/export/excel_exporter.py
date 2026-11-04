"""Excel (.xlsx) export module with multi-sheet analysis, formatting, and formulas."""

from __future__ import annotations

import io
import logging
from collections import defaultdict
from decimal import Decimal
from typing import Optional, Any
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

logger = logging.getLogger(__name__)


def export_to_excel(
    transactions: list[dict[str, Any]],
    account_details: Optional[dict[str, Any]] = None,
    quality_report: Optional[dict[str, Any]] = None,
    recurring_groups: Optional[list[dict[str, Any]]] = None,
    filename: str = "statement.pdf",
    processing_timestamp: str = "",
) -> bytes:
    """Export complete processed bank statement dataset to formatted multi-sheet Excel file.

    Sheets:
    1. Transactions (with frozen header, filters, flags)
    2. Account Summary
    3. Category Summary
    4. Monthly Summary
    5. Recurring Payments
    6. Quality Report
    """
    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # Remove default blank sheet

    # Colors and Styles
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    flag_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    border_thin = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Sheet 1: Transactions
    # ─────────────────────────────────────────────────────────────────────────
    ws_tx = wb.create_sheet(title="Transactions")
    tx_headers = [
        "Date", "Value Date", "Description", "Ref / Chq No",
        "Debit", "Credit", "Balance", "Category",
        "Confidence", "Source", "Flagged", "Flag Reasons"
    ]
    ws_tx.append(tx_headers)

    for cell in ws_tx[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for row_idx, tx in enumerate(transactions, start=2):
        debit_val = float(tx.get("debit") or 0.0) if tx.get("debit") not in (None, "") else None
        credit_val = float(tx.get("credit") or 0.0) if tx.get("credit") not in (None, "") else None
        balance_val = float(tx.get("balance") or 0.0) if tx.get("balance") not in (None, "") else None

        conf_val = tx.get("confidence")
        conf_str = f"{float(conf_val):.1%}" if conf_val is not None else ""

        is_flagged = bool(tx.get("is_flagged") or tx.get("flag_reasons"))
        flags_str = "; ".join(tx.get("flag_reasons", [])) if tx.get("flag_reasons") else ""

        row_data = [
            tx.get("date", ""),
            tx.get("value_date", ""),
            tx.get("description", ""),
            tx.get("ref_no", ""),
            debit_val,
            credit_val,
            balance_val,
            tx.get("category", "Other/Uncategorized"),
            conf_str,
            tx.get("source", "rule"),
            "YES" if is_flagged else "NO",
            flags_str,
        ]
        ws_tx.append(row_data)

        # Apply formatting
        for col_idx in range(1, len(row_data) + 1):
            cell = ws_tx.cell(row=row_idx, column=col_idx)
            cell.border = border_thin
            if is_flagged:
                cell.fill = flag_fill

            # Numbers formatting (Debit, Credit, Balance)
            if col_idx in (5, 6, 7) and cell.value is not None:
                cell.number_format = "#,##0.00"

    ws_tx.freeze_panes = "A2"
    ws_tx.auto_filter.ref = ws_tx.dimensions

    # ─────────────────────────────────────────────────────────────────────────
    # Sheet 2: Account Summary
    # ─────────────────────────────────────────────────────────────────────────
    ws_acc = wb.create_sheet(title="Account Summary")
    ws_acc.append(["Account Details", ""])
    ws_acc.append(["Source File", filename])
    ws_acc.append(["Processed At", processing_timestamp])
    ws_acc.append(["", ""])

    acc = account_details or {}
    acc_fields = [
        ("Account Holder", acc.get("account_holder_name") or acc.get("holder", "N/A")),
        ("Bank Name", acc.get("bank_name", "N/A")),
        ("Account Number", acc.get("account_number") or acc.get("acc_no", "N/A")),
        ("Account Type", acc.get("account_type") or acc.get("acc_type", "N/A")),
        ("IFSC Code", acc.get("ifsc", "N/A")),
        ("MICR Code", acc.get("micr", "N/A")),
        ("Branch", acc.get("branch", "N/A")),
        ("Customer ID", acc.get("customer_id") or acc.get("cust_id", "N/A")),
        ("Address", acc.get("address", "N/A")),
        ("Statement Period From", str(acc.get("period_from", "N/A"))),
        ("Statement Period To", str(acc.get("period_to", "N/A"))),
        ("Opening Balance", acc.get("opening_balance", "N/A")),
        ("Closing Balance", acc.get("closing_balance", "N/A")),
    ]
    for label, val in acc_fields:
        ws_acc.append([label, str(val) if val is not None else "N/A"])

    for cell in ws_acc[1]:
        cell.font = header_font
        cell.fill = header_fill

    # ─────────────────────────────────────────────────────────────────────────
    # Sheet 3: Category Summary
    # ─────────────────────────────────────────────────────────────────────────
    ws_cat = wb.create_sheet(title="Category Summary")
    ws_cat.append(["Category", "Total Debit (Spend)", "Total Credit (Income)", "Net Amount", "Txn Count", "% of Spend"])
    for cell in ws_cat[1]:
        cell.font = header_font
        cell.fill = header_fill

    cat_totals: dict[str, dict[str, float]] = defaultdict(lambda: {"debit": 0.0, "credit": 0.0, "count": 0})
    total_debit_all = 0.0

    for tx in transactions:
        c = tx.get("category", "Other/Uncategorized")
        d = float(tx.get("debit") or 0.0) if tx.get("debit") not in (None, "") else 0.0
        cr = float(tx.get("credit") or 0.0) if tx.get("credit") not in (None, "") else 0.0
        cat_totals[c]["debit"] += d
        cat_totals[c]["credit"] += cr
        cat_totals[c]["count"] += 1
        total_debit_all += d

    sorted_cats = sorted(cat_totals.items(), key=lambda x: -x[1]["debit"])
    for c_name, stats in sorted_cats:
        net = stats["credit"] - stats["debit"]
        pct = (stats["debit"] / total_debit_all) if total_debit_all > 0 else 0.0
        ws_cat.append([
            c_name,
            stats["debit"],
            stats["credit"],
            net,
            stats["count"],
            pct,
        ])

    for row in ws_cat.iter_rows(min_row=2, min_col=2, max_col=4):
        for cell in row:
            cell.number_format = "#,##0.00"
    for row in ws_cat.iter_rows(min_row=2, min_col=6, max_col=6):
        for cell in row:
            cell.number_format = "0.0%"

    # ─────────────────────────────────────────────────────────────────────────
    # Sheet 4: Monthly Summary
    # ─────────────────────────────────────────────────────────────────────────
    ws_month = wb.create_sheet(title="Monthly Summary")
    ws_month.append(["Month (YYYY-MM)", "Total Inflow (Credit)", "Total Outflow (Debit)", "Net Savings", "Txn Count"])
    for cell in ws_month[1]:
        cell.font = header_font
        cell.fill = header_fill

    monthly_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"debit": 0.0, "credit": 0.0, "count": 0})
    for tx in transactions:
        d_str = str(tx.get("date", ""))[:7]
        if len(d_str) == 7:
            d = float(tx.get("debit") or 0.0) if tx.get("debit") not in (None, "") else 0.0
            cr = float(tx.get("credit") or 0.0) if tx.get("credit") not in (None, "") else 0.0
            monthly_stats[d_str]["debit"] += d
            monthly_stats[d_str]["credit"] += cr
            monthly_stats[d_str]["count"] += 1

    for m_key in sorted(monthly_stats.keys()):
        m_data = monthly_stats[m_key]
        net = m_data["credit"] - m_data["debit"]
        ws_month.append([m_key, m_data["credit"], m_data["debit"], net, m_data["count"]])

    for row in ws_month.iter_rows(min_row=2, min_col=2, max_col=4):
        for cell in row:
            cell.number_format = "#,##0.00"

    # ─────────────────────────────────────────────────────────────────────────
    # Sheet 5: Recurring Payments
    # ─────────────────────────────────────────────────────────────────────────
    ws_rec = wb.create_sheet(title="Recurring Payments")
    ws_rec.append(["Merchant / Payee", "Category", "Frequency", "Average Amount", "Occurrences", "Last Date", "Next Expected"])
    for cell in ws_rec[1]:
        cell.font = header_font
        cell.fill = header_fill

    if recurring_groups:
        for rg in recurring_groups:
            ws_rec.append([
                rg.get("merchant", ""),
                rg.get("category", ""),
                rg.get("frequency", ""),
                float(rg.get("avg_amount", 0.0)),
                len(rg.get("transaction_indices", [])),
                rg.get("last_date", ""),
                rg.get("next_expected_date", ""),
            ])
        for row in ws_rec.iter_rows(min_row=2, min_col=4, max_col=4):
            for cell in row:
                cell.number_format = "#,##0.00"

    # ─────────────────────────────────────────────────────────────────────────
    # Sheet 6: Quality Report
    # ─────────────────────────────────────────────────────────────────────────
    ws_qr = wb.create_sheet(title="Quality Report")
    ws_qr.append(["Extraction Quality Metric", "Value"])
    for cell in ws_qr[1]:
        cell.font = header_font
        cell.fill = header_fill

    qr = quality_report or {}
    qr_metrics = [
        ("Quality Grade", qr.get("grade", "N/A")),
        ("Total Pages Processed", qr.get("total_pages", "N/A")),
        ("Native Text Pages", qr.get("text_pages", "N/A")),
        ("OCR Pages", qr.get("ocr_pages", "N/A")),
        ("Total Rows Extracted", qr.get("rows_extracted", len(transactions))),
        ("Flagged Rows", qr.get("rows_flagged", 0)),
        ("Reconciliation Pass Rate", f"{float(qr.get('reconciliation_rate', 1.0)):.1%}" if qr.get("reconciliation_rate") is not None else "N/A"),
        ("Summary Balance Valid", "YES" if qr.get("summary_balance_ok") else "NO"),
        ("Duplicate Rows Found", qr.get("duplicate_count", 0)),
        ("Balance Chain Gaps", qr.get("gap_count", 0)),
        ("Out-of-Period Dates", qr.get("out_of_period_count", 0)),
        ("Zero Amount Rows", qr.get("zero_amount_count", 0)),
        ("Auto-Corrections Applied", qr.get("auto_corrections", 0)),
        ("Warnings", "; ".join(qr.get("warnings", [])) or "None"),
    ]
    for m_label, m_val in qr_metrics:
        ws_qr.append([m_label, str(m_val)])

    # Auto-adjust column widths for all sheets
    for sheet in wb.worksheets:
        for col in sheet.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = get_column_letter(col[0].column)
            sheet.column_dimensions[col_letter].width = min(50, max(12, max_len + 3))

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

