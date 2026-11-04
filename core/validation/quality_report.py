"""Extraction quality report builder.

Aggregates signals from page-type classification, row extraction, OCR
confidence, and balance reconciliation into a single graded
:class:`QualityReport`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-class
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Quality assessment for a single processed bank statement."""

    filename: str
    total_pages: int
    text_pages: int
    ocr_pages: int
    rows_extracted: int
    rows_flagged: int
    reconciliation_rate: float          # 0.0 – 1.0
    avg_ocr_confidence: Optional[float]
    min_ocr_confidence: Optional[float]
    duplicate_count: int
    gap_count: int
    out_of_period_count: int
    zero_amount_count: int
    reversal_pair_count: int
    auto_corrections: int
    summary_balance_ok: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    grade: str = "A"                    # A / B / C / D / F

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------

    def compute_grade(self) -> str:
        """Grade extraction quality from **A** (best) to **F** (worst).

        Grading rubric
        --------------
        * **A** – reconciliation ≥ 95 %, no hard errors, no duplicates,
          avg OCR confidence ≥ 90 % (or N/A).
        * **B** – reconciliation 85–95 %, minor warnings, avg OCR ≥ 75 %.
        * **C** – reconciliation 70–85 %.
        * **D** – reconciliation 50–70 %.
        * **F** – reconciliation < 50 %, or any hard error present.

        The grade is stored back into ``self.grade`` and returned.
        """
        # Hard-fail: any error strings → F immediately
        if self.errors:
            self.grade = "F"
            return self.grade

        rate = self.reconciliation_rate
        avg_conf = self.avg_ocr_confidence if self.avg_ocr_confidence is not None else 1.0

        if rate < 0.50:
            self.grade = "F"
        elif rate < 0.70:
            self.grade = "D"
        elif rate < 0.85:
            self.grade = "C"
        elif rate < 0.95:
            # Downgrade B → C if OCR confidence is very low
            if avg_conf < 0.60:
                self.grade = "C"
            else:
                self.grade = "B"
        else:
            # ≥ 95 %
            if avg_conf < 0.75 or self.duplicate_count > 0:
                self.grade = "B"
            else:
                self.grade = "A"

        logger.debug(
            "Quality grade for %s: %s (rate=%.2f, avg_conf=%.2f)",
            self.filename,
            self.grade,
            rate,
            avg_conf,
        )
        return self.grade

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return all metrics as a plain :class:`dict`.

        Suitable for JSON serialisation or passing to export functions.
        """
        return {
            "filename": self.filename,
            "total_pages": self.total_pages,
            "text_pages": self.text_pages,
            "ocr_pages": self.ocr_pages,
            "rows_extracted": self.rows_extracted,
            "rows_flagged": self.rows_flagged,
            "reconciliation_rate": round(self.reconciliation_rate, 4),
            "avg_ocr_confidence": (
                round(self.avg_ocr_confidence, 4)
                if self.avg_ocr_confidence is not None
                else None
            ),
            "min_ocr_confidence": (
                round(self.min_ocr_confidence, 4)
                if self.min_ocr_confidence is not None
                else None
            ),
            "duplicate_count": self.duplicate_count,
            "gap_count": self.gap_count,
            "out_of_period_count": self.out_of_period_count,
            "zero_amount_count": self.zero_amount_count,
            "reversal_pair_count": self.reversal_pair_count,
            "auto_corrections": self.auto_corrections,
            "summary_balance_ok": self.summary_balance_ok,
            "warnings": self.warnings,
            "errors": self.errors,
            "grade": self.grade,
        }

    def to_markdown(self) -> str:
        """Render the quality report as a Markdown string.

        Returns a human-readable Markdown table suitable for embedding in
        reports or displaying in documentation.
        """
        grade_emoji = {
            "A": "🟢",
            "B": "🟡",
            "C": "🟠",
            "D": "🔴",
            "F": "⛔",
        }.get(self.grade, "❓")

        ocr_conf_str = (
            f"{self.avg_ocr_confidence * 100:.1f}%"
            if self.avg_ocr_confidence is not None
            else "N/A"
        )
        min_conf_str = (
            f"{self.min_ocr_confidence * 100:.1f}%"
            if self.min_ocr_confidence is not None
            else "N/A"
        )

        lines: list[str] = [
            f"# Quality Report — `{self.filename}`\n",
            f"**Grade:** {grade_emoji} **{self.grade}**\n",
            "## Extraction Metrics\n",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Total Pages | {self.total_pages} |",
            f"| Text-extracted Pages | {self.text_pages} |",
            f"| OCR-processed Pages | {self.ocr_pages} |",
            f"| Rows Extracted | {self.rows_extracted} |",
            f"| Rows Flagged | {self.rows_flagged} |",
            f"| Reconciliation Rate | {self.reconciliation_rate * 100:.1f}% |",
            f"| Avg OCR Confidence | {ocr_conf_str} |",
            f"| Min OCR Confidence | {min_conf_str} |",
            "\n## Anomaly Counts\n",
            "| Anomaly | Count |",
            "| --- | --- |",
            f"| Duplicates | {self.duplicate_count} |",
            f"| Balance Gaps | {self.gap_count} |",
            f"| Out-of-period | {self.out_of_period_count} |",
            f"| Zero-amount | {self.zero_amount_count} |",
            f"| Reversal Pairs | {self.reversal_pair_count} |",
            f"| Auto-corrections | {self.auto_corrections} |",
            f"| Summary Balance OK | {'✅ Yes' if self.summary_balance_ok else '❌ No'} |",
        ]

        if self.warnings:
            lines.append("\n## ⚠️ Warnings\n")
            for w in self.warnings:
                lines.append(f"- {w}")

        if self.errors:
            lines.append("\n## ❌ Errors\n")
            for e in self.errors:
                lines.append(f"- {e}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder function
# ---------------------------------------------------------------------------

def build_quality_report(
    filename: str,
    page_types: list[str],
    transactions: list[dict],
    reconciliation_report: Any,
    ocr_results: list[dict],
) -> QualityReport:
    """Build a :class:`QualityReport` from extraction artefacts.

    Parameters
    ----------
    filename:
        Original PDF filename (used for display only).
    page_types:
        List of page-type strings, one per page.  Expected values are
        ``"text"`` (native PDF text) or ``"ocr"`` (image-based OCR).
    transactions:
        List of extracted transaction row dicts (after classification).
    reconciliation_report:
        A :class:`~core.validation.balance_reconciler.ReconciliationReport`
        instance produced by :func:`~core.validation.balance_reconciler.reconcile`.
    ocr_results:
        List of OCR result dicts.  Each dict may contain a ``"confidence"``
        key (float, 0–1 or 0–100).

    Returns
    -------
    QualityReport
        Fully populated and graded report.
    """
    logger.info("Building quality report for '%s'", filename)

    total_pages = len(page_types)
    text_pages = sum(1 for pt in page_types if str(pt).lower() == "text")
    ocr_pages = total_pages - text_pages

    # Rows flagged = any row that has a non-empty flag_reason in reconciliation
    rows_flagged = sum(
        1
        for r in (reconciliation_report.row_results if reconciliation_report else [])
        if r.flag_reason
    )

    # OCR confidence aggregation
    confidences: list[float] = []
    for ocr in ocr_results:
        conf = ocr.get("confidence")
        if conf is not None:
            # Normalise to 0–1 range
            conf_float = float(conf)
            if conf_float > 1.0:
                conf_float /= 100.0
            confidences.append(conf_float)

    avg_ocr_confidence = (sum(confidences) / len(confidences)) if confidences else None
    min_ocr_confidence = min(confidences) if confidences else None

    # Gather flags from reconciliation report
    recon = reconciliation_report
    duplicate_count = len(recon.duplicate_indices) if recon else 0
    gap_count = len(recon.gap_indices) if recon else 0
    out_of_period_count = len(recon.out_of_period_indices) if recon else 0
    zero_amount_count = len(recon.zero_amount_indices) if recon else 0
    reversal_pair_count = len(recon.reversal_pairs) if recon else 0
    auto_corrections = recon.auto_corrected_rows if recon else 0
    reconciliation_rate = recon.reconciliation_rate if recon else 1.0
    summary_balance_ok = recon.summary_balance_ok if recon else False
    recon_warnings: list[str] = recon.warnings if recon else []

    # Collect any hard errors from transactions
    errors: list[str] = []

    report = QualityReport(
        filename=filename,
        total_pages=total_pages,
        text_pages=text_pages,
        ocr_pages=ocr_pages,
        rows_extracted=len(transactions),
        rows_flagged=rows_flagged,
        reconciliation_rate=reconciliation_rate,
        avg_ocr_confidence=avg_ocr_confidence,
        min_ocr_confidence=min_ocr_confidence,
        duplicate_count=duplicate_count,
        gap_count=gap_count,
        out_of_period_count=out_of_period_count,
        zero_amount_count=zero_amount_count,
        reversal_pair_count=reversal_pair_count,
        auto_corrections=auto_corrections,
        summary_balance_ok=summary_balance_ok,
        warnings=recon_warnings,
        errors=errors,
    )

    report.compute_grade()
    logger.info(
        "Quality report built for '%s': grade=%s, rate=%.1f%%",
        filename,
        report.grade,
        reconciliation_rate * 100,
    )
    return report
