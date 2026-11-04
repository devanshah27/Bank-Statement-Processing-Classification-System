"""Comprehensive dataset evaluation and benchmarking script.

Runs the extraction and classification pipeline over every PDF in the test dataset,
compares against ground truth (exact date + debit + credit + balance matches,
whitespace-insensitive description match, and account details match),
and writes Markdown and CSV benchmark reports to reports/.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline import StatementPipeline
from core.export.csv_exporter import export_to_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def num(x: Any) -> Decimal:
    """Normalize string/float numeric value to Decimal for comparison."""
    if x is None or str(x).strip() in ("", "-", "NIL"):
        return Decimal(0)
    cleaned = str(x).replace(",", "").strip()
    try:
        return Decimal(cleaned)
    except Exception:
        return Decimal(0)


def desc_norm(d: Any) -> str:
    """Normalize description for whitespace-insensitive comparison."""
    if not d:
        return ""
    return " ".join(str(d).split()).strip().lower()


def score_transactions(gt_rows: list[dict[str, Any]], extracted_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score extracted transactions against ground truth rows."""
    def key_tuple(r):
        return (
            str(r.get("date", "")).strip(),
            num(r.get("debit")),
            num(r.get("credit")),
            num(r.get("balance")),
        )

    gt_keys = [key_tuple(r) for r in gt_rows]
    out_keys = [key_tuple(r) for r in extracted_rows]

    pool = list(out_keys)
    hit = 0
    for k in gt_keys:
        if k in pool:
            pool.remove(k)
            hit += 1

    gt_count = len(gt_rows)
    out_count = len(extracted_rows)
    exact_match_pct = (hit / max(1, gt_count)) * 100.0

    # Also compute description match rate among matched or aligned rows
    desc_hits = 0
    compare_len = min(gt_count, out_count)
    for i in range(compare_len):
        if desc_norm(gt_rows[i].get("description")) == desc_norm(extracted_rows[i].get("description")):
            desc_hits += 1
    desc_match_pct = (desc_hits / max(1, gt_count)) * 100.0

    return {
        "gt_rows": gt_count,
        "extracted_rows": out_count,
        "exact_matches": hit,
        "exact_match_pct": exact_match_pct,
        "missing_rows": gt_count - hit,
        "extra_rows": len(pool),
        "desc_match_pct": desc_match_pct,
    }


def score_account_details(gt_account: dict[str, Any], extracted_account: dict[str, Any]) -> dict[str, Any]:
    """Score extracted account details against ground truth json."""
    gt_acc = gt_account.get("account", gt_account)
    ext_acc = extracted_account

    fields_to_check = ["holder", "acc_no", "ifsc", "bank_name", "branch", "cust_id"]
    matches = 0
    checked = 0

    for f in fields_to_check:
        gt_val = gt_acc.get(f) or gt_acc.get(f"account_{f}")
        if gt_val:
            checked += 1
            ext_val = ext_acc.get(f) or ext_acc.get(f"account_{f}")
            if ext_val and str(gt_val).strip().lower() in str(ext_val).strip().lower():
                matches += 1

    pct = (matches / max(1, checked)) * 100.0 if checked > 0 else 100.0
    return {"checked_fields": checked, "matched_fields": matches, "match_pct": pct}


def evaluate_all(
    dataset_dir: Path,
    output_dir: Path,
    ocr_engine: str = "rapid",
) -> dict[str, Any]:
    """Run full benchmark over all dataset categories."""
    pipeline = StatementPipeline(ocr_engine_name=ocr_engine)
    ground_truth_dir = dataset_dir / "ground_truth"

    categories = [
        ("text_based", "Native Text PDF Layouts"),
        ("scanned_image_based", "Scanned & Degraded Image PDFs"),
        ("mixed_and_multi", "Mixed Pages & Multi-Account Statements"),
        ("edge_cases", "Planted Defects & File Corruptions"),
        ("other_document_types", "Non-Statement Documents"),
    ]

    benchmark_results: list[dict[str, Any]] = []

    for cat_folder, cat_title in categories:
        folder_path = dataset_dir / cat_folder
        if not folder_path.is_dir():
            continue

        pdf_files = sorted(list(folder_path.glob("*.pdf")))
        for pdf_path in pdf_files:
            stem = pdf_path.stem
            logger.info("Evaluating [%s] %s...", cat_folder, pdf_path.name)

            gt_csv = ground_truth_dir / f"{stem}.csv"
            gt_json = ground_truth_dir / f"{stem}.json"

            gt_rows: list[dict[str, Any]] = []
            if gt_csv.is_file():
                with open(gt_csv, "r", encoding="utf-8") as f:
                    gt_rows = list(csv.DictReader(f))

            gt_acc: dict[str, Any] = {}
            if gt_json.is_file():
                with open(gt_json, "r", encoding="utf-8") as f:
                    gt_acc = json.load(f)

            # Special case for password protected
            password = "test1234" if "password_protected" in stem else None

            start_t = time.time()
            try:
                res = pipeline.process(pdf_path.read_bytes(), pdf_path.name, password=password)
                duration = time.time() - start_t
            except Exception as e:
                logger.exception("Processing error on %s", pdf_path.name)
                duration = time.time() - start_t
                benchmark_results.append({
                    "category": cat_folder,
                    "filename": pdf_path.name,
                    "status": "CRASH",
                    "duration_sec": duration,
                    "gt_rows": len(gt_rows),
                    "extracted_rows": 0,
                    "exact_matches": 0,
                    "exact_match_pct": 0.0,
                    "desc_match_pct": 0.0,
                    "account_match_pct": 0.0,
                    "notes": str(e),
                })
                continue

            if not res.success:
                benchmark_results.append({
                    "category": cat_folder,
                    "filename": pdf_path.name,
                    "status": "REJECTED_CLEANLY",
                    "duration_sec": duration,
                    "gt_rows": len(gt_rows),
                    "extracted_rows": 0,
                    "exact_matches": 0,
                    "exact_match_pct": 100.0 if not gt_rows else 0.0,
                    "desc_match_pct": 0.0,
                    "account_match_pct": 0.0,
                    "notes": f"Code: {res.error_code} ({res.error_message})",
                })
                continue

            tx_score = score_transactions(gt_rows, res.transactions)
            acc_score = score_account_details(gt_acc, res.account)

            benchmark_results.append({
                "category": cat_folder,
                "filename": pdf_path.name,
                "status": "SUCCESS",
                "duration_sec": duration,
                "gt_rows": tx_score["gt_rows"],
                "extracted_rows": tx_score["extracted_rows"],
                "exact_matches": tx_score["exact_matches"],
                "exact_match_pct": tx_score["exact_match_pct"],
                "desc_match_pct": tx_score["desc_match_pct"],
                "account_match_pct": acc_score["match_pct"],
                "notes": f"Grade: {res.quality.get('grade')} | Recon: {res.quality.get('reconciliation_rate', 0):.1%}",
            })

    # Generate Reports
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_report_path = output_dir / f"benchmark_{ocr_engine}_{timestamp}.csv"
    md_report_path = output_dir / f"benchmark_{ocr_engine}_{timestamp}.md"

    # 1. CSV Report
    with open(csv_report_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "category", "filename", "status", "duration_sec", "gt_rows",
            "extracted_rows", "exact_matches", "exact_match_pct",
            "desc_match_pct", "account_match_pct", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(benchmark_results)

    # 2. Markdown Report
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(f"# Bank Statement Extraction Benchmark Report\n\n")
        f.write(f"- **Timestamp**: {datetime.datetime.now().isoformat()}\n")
        f.write(f"- **OCR Engine**: `{ocr_engine}`\n")
        f.write(f"- **Total Documents Tested**: {len(benchmark_results)}\n\n")

        # Summary by category
        f.write("## Category Summary\n\n")
        f.write("| Category | Files | Total GT Rows | Extracted Rows | Exact Matches | Match Rate | Avg Duration |\n")
        f.write("|---|---|---|---|---|---|---|\n")

        cat_grouped: dict[str, list[dict[str, Any]]] = {}
        for r in benchmark_results:
            cat_grouped.setdefault(r["category"], []).append(r)

        for cat, items in cat_grouped.items():
            tot_gt = sum(i["gt_rows"] for i in items)
            tot_ext = sum(i["extracted_rows"] for i in items)
            tot_matches = sum(i["exact_matches"] for i in items)
            overall_pct = (tot_matches / max(1, tot_gt)) * 100.0 if tot_gt > 0 else 100.0
            avg_dur = sum(i["duration_sec"] for i in items) / len(items)
            f.write(f"| **{cat}** | {len(items)} | {tot_gt} | {tot_ext} | {tot_matches} | **{overall_pct:.1f}%** | {avg_dur:.2f}s |\n")

        f.write("\n## Detailed Per-File Benchmark\n\n")
        f.write("| File | Category | Status | Time | GT Rows | Extracted | Exact Match % | Notes |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in benchmark_results:
            f.write(
                f"| `{r['filename']}` | {r['category']} | {r['status']} | {r['duration_sec']:.2f}s | "
                f"{r['gt_rows']} | {r['extracted_rows']} | **{r['exact_match_pct']:.1f}%** | {r['notes']} |\n"
            )

    logger.info("Saved benchmark reports to %s and %s", csv_report_path, md_report_path)
    return {"csv_report": csv_report_path, "md_report": md_report_path}


def main():
    parser = argparse.ArgumentParser(description="Evaluate bank statement extractor on full test dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("samples/test_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--ocr-engine", type=str, default="rapid", choices=["rapid", "tesseract", "both"])
    args = parser.parse_args()

    if args.ocr_engine == "both":
        logger.info("=== Running Benchmark with RapidOCR ===")
        evaluate_all(args.dataset_dir, args.output_dir, ocr_engine="rapid")
        logger.info("=== Running Benchmark with Tesseract ===")
        evaluate_all(args.dataset_dir, args.output_dir, ocr_engine="tesseract")
    else:
        evaluate_all(args.dataset_dir, args.output_dir, ocr_engine=args.ocr_engine)


if __name__ == "__main__":
    main()

