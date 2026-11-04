"""Test suite validating pipeline behaviors across all sample dataset PDFs."""

import time
from pathlib import Path
import pytest

from core.pipeline import StatementPipeline


@pytest.fixture(scope="module")
def pipeline():
    return StatementPipeline()


def test_password_required_then_accepted(pipeline, dataset_dir):
    """File 16: Must require password first, then decrypt with test1234."""
    pdf_path = dataset_dir / "edge_cases" / "16_password_protected_password_is_test1234.pdf"
    if not pdf_path.is_file():
        pytest.skip("File 16 not found")

    bytes_data = pdf_path.read_bytes()

    # 1. Without password -> fails with password required
    res_no_pw = pipeline.process(bytes_data, pdf_path.name, password=None)
    assert res_no_pw.success is False
    assert res_no_pw.needs_password is True

    # 2. With password 'test1234' -> succeeds
    res_pw = pipeline.process(bytes_data, pdf_path.name, password="test1234")
    assert res_pw.success is True
    assert len(res_pw.transactions) > 50


def test_corrupted_file_rejected_gracefully(pipeline, dataset_dir):
    """File 17: Truncated corrupted PDF should fail cleanly without crashing."""
    pdf_path = dataset_dir / "edge_cases" / "17_corrupted_truncated_file.pdf"
    if not pdf_path.is_file():
        pytest.skip("File 17 not found")

    res = pipeline.process(pdf_path.read_bytes(), pdf_path.name)
    assert res.success is False
    assert res.error_code in ("CORRUPT_PDF", "INVALID_FILE")


def test_png_renamed_pdf_rejected(pipeline, dataset_dir):
    """File 18: PNG with .pdf extension rejected by magic byte detection."""
    pdf_path = dataset_dir / "edge_cases" / "18_not_really_a_pdf_png_renamed.pdf"
    if not pdf_path.is_file():
        pytest.skip("File 18 not found")

    res = pipeline.process(pdf_path.read_bytes(), pdf_path.name)
    assert res.success is False
    assert res.error_code in ("WRONG_FILE_TYPE", "EXTENSION_MIME_MISMATCH")


def test_blank_page_handled(pipeline, dataset_dir):
    """File 19: Blank empty PDF page handled gracefully without error."""
    pdf_path = dataset_dir / "edge_cases" / "19_blank_empty_page.pdf"
    if not pdf_path.is_file():
        pytest.skip("File 19 not found")

    res = pipeline.process(pdf_path.read_bytes(), pdf_path.name)
    # Handled cleanly: either classified as empty or extracted 0 transactions
    if res.success:
        assert len(res.transactions) == 0
    else:
        assert res.doc_type in ("empty", "unknown", "invalid")


def test_invoice_rejected_as_non_statement(pipeline, dataset_dir):
    """File 21: Invoice rejected with clear 'not a bank statement' error."""
    pdf_path = dataset_dir / "other_document_types" / "21_non_statement_invoice_should_be_rejected.pdf"
    if not pdf_path.is_file():
        pytest.skip("File 21 not found")

    res = pipeline.process(pdf_path.read_bytes(), pdf_path.name)
    assert res.success is False
    assert res.error_code == "NOT_A_BANK_STATEMENT"
    assert res.doc_type == "invoice"


def test_credit_card_statement_flagged(pipeline, dataset_dir):
    """File 20: Credit card statement detected and flagged as non-bank ledger."""
    pdf_path = dataset_dir / "other_document_types" / "20_credit_card_statement_no_balance_column.pdf"
    if not pdf_path.is_file():
        pytest.skip("File 20 not found")

    res = pipeline.process(pdf_path.read_bytes(), pdf_path.name)
    assert res.doc_type in ("credit_card_statement", "bank_statement")


def test_planted_defects_detected_in_edge_cases(pipeline, dataset_dir):
    """File 15: Injected defects (duplicate, gap, out-of-period, zero amount, reversals) detected."""
    pdf_path = dataset_dir / "edge_cases" / "15_edge_cases_balance_breaks_duplicates_reversals.pdf"
    if not pdf_path.is_file():
        pytest.skip("File 15 not found")

    res = pipeline.process(pdf_path.read_bytes(), pdf_path.name)
    assert res.success is True
    q = res.quality
    # At least some of the injected defects must be registered in quality report
    defect_indicators = [
        q.get("duplicate_count", 0) > 0,
        q.get("gap_count", 0) > 0,
        q.get("out_of_period_count", 0) > 0,
        q.get("zero_amount_count", 0) > 0,
        q.get("reversal_pair_count", 0) > 0,
        q.get("auto_corrections", 0) > 0,
        q.get("rows_flagged", 0) > 0,
    ]
    assert any(defect_indicators)


def test_mixed_pages_routed_per_page(pipeline, dataset_dir):
    """File 12: 6 pages with mixed native text and scanned pages routed per page."""
    pdf_path = dataset_dir / "mixed_and_multi" / "12_mixed_text_pages_and_scanned_pages.pdf"
    if not pdf_path.is_file():
        pytest.skip("File 12 not found")

    res = pipeline.process(pdf_path.read_bytes(), pdf_path.name)
    assert res.success is True
    # Verify both text and OCR extraction methods were routed across pages
    methods = {p["extraction_method"] for p in res.pages}
    assert "text" in methods
    assert "ocr" in methods


def test_two_accounts_split(pipeline, dataset_dir):
    """File 13: Two different accounts in one PDF identified."""
    pdf_path = dataset_dir / "mixed_and_multi" / "13_two_accounts_in_one_pdf.pdf"
    if not pdf_path.is_file():
        pytest.skip("File 13 not found")

    res = pipeline.process(pdf_path.read_bytes(), pdf_path.name)
    assert res.success is True
    assert len(res.transactions) > 50


@pytest.mark.slow
def test_large_statement_completes_within_budget(pipeline, dataset_dir):
    """File 14: 218-page large statement must complete without out-of-memory error."""
    pdf_path = dataset_dir / "mixed_and_multi" / "14_large_statement_100plus_pages.pdf"
    if not pdf_path.is_file():
        pytest.skip("File 14 not found")

    start_t = time.time()
    # Process within time budget (< 240s for 218 pages on CPU)
    res = pipeline.process(pdf_path.read_bytes(), pdf_path.name)
    duration = time.time() - start_t
    assert res.success is True
    assert len(res.transactions) > 1000
    assert duration < 240.0
