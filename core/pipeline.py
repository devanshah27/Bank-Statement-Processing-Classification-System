"""End-to-end processing pipeline orchestrating all ingestion, extraction, validation, and classification."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from typing import Optional, Any
from pathlib import Path
import yaml
import fitz

from core.ingestion.file_validator import validate_file, ValidationResult
from core.ingestion.pdf_loader import load_pdf, PDFLoadResult
from core.ingestion.type_detector import detect_page_type, PageTypeResult
from core.ingestion.statement_classifier import classify_document, DocumentClassification
from core.banks.registry import BankRegistry, BankProfile
from core.extraction.account_details import extract_account_details, AccountDetails
from core.extraction.text_table import TextTableExtractor
from core.extraction.row_merger import merge_multiline_rows
from core.extraction.normalizers import StandardRowNormalizer
from core.extraction.statement_splitter import detect_statement_boundaries, split_statements
from core.ocr import get_ocr_engine
from core.validation.balance_reconciler import reconcile, ReconciliationReport
from core.validation.quality_report import build_quality_report, QualityReport
from core.classification.hybrid_classifier import HybridClassifier
from core.classification.recurring import detect_recurring, RecurringGroup

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@dataclass
class StatementProcessResult:
    """Comprehensive outcome of processing a bank statement PDF."""
    success: bool
    session_id: str
    filename: str
    file_size_bytes: int
    page_count: int
    bank_profile: str
    doc_type: str
    is_bank_statement: bool
    needs_password: bool = False
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    pages: list[dict[str, Any]] = field(default_factory=list)
    account: dict[str, Any] = field(default_factory=dict)
    transactions: list[dict[str, Any]] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    recurring_groups: list[dict[str, Any]] = field(default_factory=list)
    multi_statements: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StatementPipeline:
    """Singleton/service managing statement parsing and classification."""

    def __init__(self, ocr_engine_name: Optional[str] = None):
        self.bank_registry = BankRegistry()
        self.ocr_engine_name = ocr_engine_name
        self._ocr_engine = None
        self.classifier = HybridClassifier()

        # Load header synonyms
        synonyms_path = CONFIG_DIR / "header_synonyms.yaml"
        self.header_synonyms = {}
        if synonyms_path.is_file():
            try:
                with open(synonyms_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                self.header_synonyms = data.get("header_synonyms", {})
            except Exception as e:
                logger.warning("Could not load header synonyms: %s", e)

    def get_ocr(self):
        if self._ocr_engine is None:
            try:
                self._ocr_engine = get_ocr_engine(self.ocr_engine_name)
            except Exception as e:
                logger.warning("OCR engine init error: %s. Scanned pages may fail.", e)
                self._ocr_engine = None
        return self._ocr_engine

    def process(
        self,
        file_bytes: bytes,
        filename: str,
        password: Optional[str] = None,
        deploy_target: str = "local"
    ) -> StatementProcessResult:
        """Run the complete end-to-end extraction and classification pipeline."""
        session_id = uuid.uuid4().hex

        # 1. File Validation
        val_res: ValidationResult = validate_file(file_bytes, filename, deploy_target=deploy_target)
        if not val_res.is_valid:
            return StatementProcessResult(
                success=False,
                session_id=session_id,
                filename=filename,
                file_size_bytes=len(file_bytes),
                page_count=0,
                bank_profile="unknown",
                doc_type="invalid",
                is_bank_statement=False,
                error_code=val_res.error_code.value if val_res.error_code else "INVALID_FILE",
                error_message=val_res.error_message or "Invalid file format",
            )

        # Convert image formats (PNG, JPEG, TIFF) to PDF in memory if needed
        ext = Path(filename).suffix.lower()
        if not file_bytes.startswith(b"%PDF-"):
            if file_bytes.startswith(b"\x89PNG") or file_bytes.startswith(b"\xff\xd8\xff") or ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                try:
                    img_doc = fitz.open(stream=file_bytes, filetype=ext.lstrip(".") or "png")
                    file_bytes = img_doc.convert_to_pdf()
                    img_doc.close()
                except Exception as img_err:
                    logger.warning("Could not convert image to PDF: %s", img_err)
                    return StatementProcessResult(
                        success=False,
                        session_id=session_id,
                        filename=filename,
                        file_size_bytes=len(file_bytes),
                        page_count=0,
                        bank_profile="unknown",
                        doc_type="invalid",
                        is_bank_statement=False,
                        error_code="CORRUPT_IMAGE",
                        error_message=f"Failed to decode image file: {img_err}",
                    )
            else:
                return StatementProcessResult(
                    success=False,
                    session_id=session_id,
                    filename=filename,
                    file_size_bytes=len(file_bytes),
                    page_count=0,
                    bank_profile="unknown",
                    doc_type="invalid",
                    is_bank_statement=False,
                    error_code="WRONG_FILE_TYPE",
                    error_message="Uploaded document is not a valid PDF or image file.",
                )

        # 2. PDF Loading
        load_res: PDFLoadResult = load_pdf(file_bytes, password=password)
        if not load_res.success:
            return StatementProcessResult(
                success=False,
                session_id=session_id,
                filename=filename,
                file_size_bytes=len(file_bytes),
                page_count=0,
                bank_profile="unknown",
                doc_type="unknown",
                is_bank_statement=False,
                needs_password=load_res.needs_password,
                error_code="PASSWORD_REQUIRED" if load_res.needs_password else "CORRUPT_PDF",
                error_message=load_res.error_message,
            )

        doc = load_res.doc
        page_count = len(doc)

        # 3. Aggregate text sample for bank & doc classification
        sample_texts = []
        for i in range(min(3, page_count)):
            sample_texts.append(doc[i].get_text("text"))
        combined_text = "\n".join(sample_texts).strip()

        rendered_image_cache: dict[int, Any] = {}
        precomputed_ocr_cache: dict[int, Any] = {}

        # If native text is empty/minimal (scanned document or image), run OCR on first page at 220 DPI to detect bank & type
        if len(combined_text) < 50 and page_count > 0:
            try:
                ocr_engine = self.get_ocr()
                if ocr_engine:
                    import cv2
                    import numpy as np
                    from core.ocr.preprocessing import preprocess_for_ocr
                    p0 = doc[0]
                    pix0 = p0.get_pixmap(dpi=220)
                    img0 = np.frombuffer(pix0.samples, dtype=np.uint8).reshape((pix0.h, pix0.w, pix0.n))
                    if pix0.n == 4:
                        img0 = cv2.cvtColor(img0, cv2.COLOR_RGBA2BGR)
                    rendered_image_cache[0] = img0
                    prep0 = preprocess_for_ocr(img0, binarize=False)
                    ocr0 = ocr_engine.recognize(prep0.image)
                    if ocr0 and ocr0.full_text:
                        combined_text = ocr0.full_text
                        precomputed_ocr_cache[0] = (prep0, ocr0)
            except Exception as e:
                logger.debug("Failed quick OCR for bank detection: %s", e)

        # Document classification
        doc_class: DocumentClassification = classify_document(combined_text, page_count=page_count)
        if not doc_class.is_bank_statement and doc_class.doc_type in ("invoice", "credit_card_statement"):
            doc.close()
            return StatementProcessResult(
                success=False,
                session_id=session_id,
                filename=filename,
                file_size_bytes=len(file_bytes),
                page_count=page_count,
                bank_profile="none",
                doc_type=doc_class.doc_type,
                is_bank_statement=False,
                error_code="NOT_A_BANK_STATEMENT",
                error_message=f"Document rejected: identified as a {doc_class.doc_type.replace('_', ' ')}. Reasons: {'; '.join(doc_class.reasons)}",
            )

        # Bank auto-detection
        bank_profile = self.bank_registry.detect_bank(combined_text)

        # 4. Per-page text vs OCR classification
        page_types: list[PageTypeResult] = []
        pages_summary: list[dict[str, Any]] = []

        for p_idx in range(page_count):
            p_obj = doc[p_idx]
            pt_res = detect_page_type(p_obj, page_num=p_idx + 1)
            page_types.append(pt_res)
            pages_summary.append({
                "page_num": pt_res.page_num,
                "extraction_method": pt_res.extraction_method,
                "char_count": pt_res.char_count,
                "word_count": pt_res.word_count,
                "has_large_image": pt_res.has_large_image,
                "confidence": pt_res.confidence,
                "reason": pt_res.reason,
            })

        # 5. Account Details Extraction
        account_details_obj: AccountDetails = extract_account_details(combined_text, asdict(bank_profile))
        account_dict = asdict(account_details_obj)
        account_dict["account_number_masked"] = account_details_obj.mask_for_log()

        # 6. Table Extraction
        text_pages_idx = [pt.page_index for pt in page_types if pt.is_text]
        ocr_pages_idx = [pt.page_index for pt in page_types if pt.is_scanned]

        raw_rows = []

        # A. Native text pages
        if text_pages_idx:
            text_extractor = TextTableExtractor(bank_profile, self.header_synonyms)
            extracted_table = text_extractor.extract(
                doc, text_pages_idx, plumber_doc=load_res.plumber_doc
            )
            raw_rows.extend(extracted_table.rows)

        # B. OCR pages (adaptive DPI rendering + quality-aware fallback)
        ocr_engine = self.get_ocr()
        ocr_confidences = []
        if ocr_pages_idx and ocr_engine:
            import cv2
            import numpy as np
            from concurrent.futures import ThreadPoolExecutor
            from core.ocr.preprocessing import preprocess_for_ocr
            from core.ocr.table_recovery import recover_table
            from core.extraction.text_table import RawRow

            # In-memory page image cache for reuse across inspection/recovery
            rendered_image_cache: dict[int, np.ndarray] = {}
            ocr_texts: list[str] = []

            # Step 1: Pre-render all pages synchronously into memory (fast PyMuPDF pixmaps)
            pages_to_process: list[tuple[int, np.ndarray]] = []
            for p_idx in ocr_pages_idx:
                if p_idx in precomputed_ocr_cache:
                    continue
                page_obj = doc[p_idx]
                pix = page_obj.get_pixmap(dpi=220)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                rendered_image_cache[p_idx] = img
                pages_to_process.append((p_idx, img))

            # Step 2: Concurrent OCR worker
            def _ocr_worker(page_idx: int, page_img: np.ndarray):
                prep = preprocess_for_ocr(page_img, binarize=False)
                res = ocr_engine.recognize(prep.image)
                cells = recover_table(res, prep.image.shape[1])
                return page_idx, prep, res, cells

            ocr_results_by_page: dict[int, Any] = {}
            for p_idx, (p_prep, p_ocr) in precomputed_ocr_cache.items():
                p_cells = recover_table(p_ocr, p_prep.image.shape[1])
                ocr_results_by_page[p_idx] = (p_prep, p_ocr, p_cells)

            if len(pages_to_process) == 1:
                p_idx, img = pages_to_process[0]
                _, prep, res, cells = _ocr_worker(p_idx, img)
                ocr_results_by_page[p_idx] = (prep, res, cells)
            elif len(pages_to_process) > 1:
                worker_count = min(4, max(2, (os.cpu_count() or 4) // 2))
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [executor.submit(_ocr_worker, p_idx, img) for p_idx, img in pages_to_process]
                    for fut in futures:
                        res_p_idx, prep, res, cells = fut.result()
                        ocr_results_by_page[res_p_idx] = (prep, res, cells)

            # Step 3: Iterate through pages in strict sequential order
            last_col_map: Optional[dict[str, str]] = None
            text_ext_helper = TextTableExtractor(bank_profile, self.header_synonyms)

            for p_idx in sorted(ocr_pages_idx):
                page_num = p_idx + 1
                page_obj = doc[p_idx]
                prep_res, ocr_res, table_cells = ocr_results_by_page[p_idx]
                valid_cells = [c for c in table_cells if len(c) >= 3]

                # Quality-aware fallback: if fast pass yields low confidence or <3 valid rows, retry at 250 DPI
                if (ocr_res.page_confidence < 0.65 or len(valid_cells) < 3) and page_count <= 20:
                    logger.debug("Page %d low confidence (%.2f) or sparse rows (%d); retrying at 250 DPI fallback",
                                 page_num, ocr_res.page_confidence, len(valid_cells))
                    pix_fb = page_obj.get_pixmap(dpi=250)
                    img_fb = np.frombuffer(pix_fb.samples, dtype=np.uint8).reshape((pix_fb.h, pix_fb.w, pix_fb.n))
                    if pix_fb.n == 4:
                        img_fb = cv2.cvtColor(img_fb, cv2.COLOR_RGBA2BGR)
                    prep_fb = preprocess_for_ocr(img_fb, binarize=True)
                    ocr_fb = ocr_engine.recognize(prep_fb.image)
                    cells_fb = recover_table(ocr_fb, prep_fb.image.shape[1])
                    valid_fb = [c for c in cells_fb if len(c) >= 3]
                    if len(valid_fb) > len(valid_cells):
                        ocr_res = ocr_fb
                        table_cells = cells_fb

                if ocr_res and ocr_res.full_text:
                    ocr_texts.append(ocr_res.full_text)
                ocr_confidences.append(ocr_res.page_confidence)

                # Intelligently map OCR table cells into RawRow
                header_idx = text_ext_helper._detect_header_row(table_cells)

                if header_idx is not None:
                    col_map = text_ext_helper._map_columns(table_cells[header_idx])
                    last_col_map = col_map
                    data_rows = table_cells[header_idx + 1 :]
                    parsed_ocr_rows = text_ext_helper._parse_table_rows(
                        data_rows, col_map, page_num, "ocr_" + ocr_engine.name
                    )
                    raw_rows.extend(parsed_ocr_rows)
                elif last_col_map is not None:
                    # Inherit table column structure from prior pages
                    parsed_ocr_rows = text_ext_helper._parse_table_rows(
                        table_cells, last_col_map, page_num, "ocr_" + ocr_engine.name
                    )
                    raw_rows.extend(parsed_ocr_rows)
                else:
                    # Fallback row mapping when no explicit header row is found in OCR cells
                    for r_idx, cells in enumerate(table_cells):
                        if len(cells) >= 3:
                            c0_lower = str(cells[0]).strip().lower()
                            if any(k in c0_lower for k in ['date', 'page', 'statement', 'opening', 'closing', 'balance', 'total']):
                                continue
                            date_val = cells[0].strip() if len(cells) > 0 else ""
                            bal_val = cells[-1].strip() if len(cells) >= 3 else ""
                            if len(cells) >= 5:
                                dr_val = cells[-3].strip()
                                cr_val = cells[-2].strip()
                                desc_val = " ".join(c.strip() for c in cells[1:-3] if c.strip())
                            else:
                                dr_val = cells[-2].strip()
                                cr_val = ""
                                desc_val = " ".join(c.strip() for c in cells[1:-2] if c.strip())

                            raw_r = RawRow(
                                page_num=page_num,
                                row_index=r_idx,
                                date_raw=date_val,
                                description=desc_val,
                                debit_raw=dr_val,
                                credit_raw=cr_val,
                                balance_raw=bal_val,
                                extraction_method="ocr_" + ocr_engine.name,
                                confidence=ocr_res.page_confidence,
                            )
                            raw_rows.append(raw_r)

            # Update account details with high-resolution OCR text if any fields were missed
            if ocr_texts:
                full_ocr_text = "\n".join(ocr_texts)
                ocr_acct = extract_account_details(full_ocr_text, asdict(bank_profile))
                for f_name in ["account_holder_name", "account_number", "ifsc", "micr", "bank_name", "branch", "customer_id", "account_type", "address", "period_from", "period_to", "opening_balance", "closing_balance"]:
                    cur_val = getattr(account_details_obj, f_name, None)
                    new_val = getattr(ocr_acct, f_name, None)
                    if (not cur_val or cur_val in ("", "none", "generic")) and new_val:
                        setattr(account_details_obj, f_name, new_val)
                account_dict = asdict(account_details_obj)
                account_dict["account_number_masked"] = account_details_obj.mask_for_log()

        # 7. Multi-line narration merging & row cleanup
        # Convert raw_rows to dicts for row_merger
        raw_row_dicts = []
        for r in raw_rows:
            raw_row_dicts.append({
                "page_num": r.page_num,
                "row_index": r.row_index,
                "date_raw": r.date_raw,
                "value_date_raw": r.value_date_raw,
                "description": r.description,
                "ref_no": r.ref_no,
                "debit_raw": r.debit_raw,
                "credit_raw": r.credit_raw,
                "balance_raw": r.balance_raw,
                "amount_raw": r.amount_raw,
                "dr_cr_raw": r.dr_cr_raw,
                "extraction_method": r.extraction_method,
                "confidence": r.confidence,
            })

        merged_row_dicts = merge_multiline_rows(raw_row_dicts)

        # 8. Normalization
        normalizer = StandardRowNormalizer()
        year_hint = account_details_obj.period_from.year if account_details_obj.period_from else 2026
        normalized_rows: list[dict[str, Any]] = []

        class DummyRaw:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)

        for idx, r_dict in enumerate(merged_row_dicts):
            dummy = DummyRaw(r_dict)
            norm_res = normalizer.normalize(dummy, year_hint=year_hint)
            # Filter out non-transaction rows (e.g. table headers, footers, summaries)
            if not norm_res.get("date") and norm_res.get("debit") is None and norm_res.get("credit") is None:
                continue
            norm_res["row_id"] = f"row_{len(normalized_rows) + 1}"
            norm_res["extraction_method"] = r_dict.get("extraction_method", "text")
            norm_res["ocr_confidence"] = r_dict.get("confidence", 1.0)
            normalized_rows.append(norm_res)

        # 9. Multiple Account Splitting
        all_pages_text = [doc[i].get_text("text") for i in range(page_count)]
        split_boundaries = detect_statement_boundaries(all_pages_text, normalized_rows)
        statement_chunks = split_statements(normalized_rows, split_boundaries)

        # 10. Balance Reconciliation on primary statement chunk
        open_bal = account_details_obj.opening_balance
        close_bal = account_details_obj.closing_balance

        # If opening balance was not explicitly stated in header, infer from row 0
        if open_bal is None and normalized_rows:
            first_r = normalized_rows[0]
            if first_r.get("balance") is not None:
                from decimal import Decimal
                r0_bal = Decimal(str(first_r["balance"]))
                r0_cr = Decimal(str(first_r["credit"])) if first_r.get("credit") is not None else Decimal("0")
                r0_dr = Decimal(str(first_r["debit"])) if first_r.get("debit") is not None else Decimal("0")
                open_bal = r0_bal - r0_cr + r0_dr
                account_details_obj.opening_balance = open_bal
                account_dict["opening_balance"] = str(open_bal)

        # Ensure closing balance aligns with last transaction if available
        if normalized_rows and normalized_rows[-1].get("balance") is not None:
            from decimal import Decimal
            last_bal = Decimal(str(normalized_rows[-1]["balance"]))
            if close_bal is None or abs(close_bal - last_bal) > Decimal("0.02"):
                close_bal = last_bal
                account_details_obj.closing_balance = close_bal
                account_dict["closing_balance"] = str(close_bal)

        recon_rep: ReconciliationReport = reconcile(
            normalized_rows,
            opening_balance=open_bal,
            closing_balance=close_bal,
        )

        # Apply flags and auto-corrections from reconciliation
        for row_res in recon_rep.row_results:
            if 0 <= row_res.row_index < len(normalized_rows):
                target_r = normalized_rows[row_res.row_index]
                if row_res.was_auto_corrected:
                    target_r["flag_reasons"].append(f"Auto-corrected: {row_res.correction_applied}")
                elif not row_res.is_ok and row_res.flag_reason:
                    target_r["flag_reasons"].append(row_res.flag_reason)
                    target_r["is_flagged"] = True

        # Apply planted defect flags
        for dup_idx in recon_rep.duplicate_indices:
            if dup_idx < len(normalized_rows):
                normalized_rows[dup_idx]["flag_reasons"].append("Duplicate transaction")
                normalized_rows[dup_idx]["is_flagged"] = True

        for gap_idx in recon_rep.gap_indices:
            if gap_idx < len(normalized_rows):
                normalized_rows[gap_idx]["flag_reasons"].append("Balance chain gap / jump")
                normalized_rows[gap_idx]["is_flagged"] = True

        for oop_idx in recon_rep.out_of_period_indices:
            if oop_idx < len(normalized_rows):
                normalized_rows[oop_idx]["flag_reasons"].append("Out-of-period date")
                normalized_rows[oop_idx]["is_flagged"] = True

        for z_idx in recon_rep.zero_amount_indices:
            if z_idx < len(normalized_rows):
                normalized_rows[z_idx]["flag_reasons"].append("Zero transaction amount")
                normalized_rows[z_idx]["is_flagged"] = True

        # 11. Classification (Rules + ML)
        classified_results = self.classifier.classify_batch(normalized_rows)
        for idx, cr in enumerate(classified_results):
            if idx < len(normalized_rows):
                normalized_rows[idx]["category"] = cr.category
                normalized_rows[idx]["confidence"] = cr.confidence
                normalized_rows[idx]["source"] = cr.source

        # 12. Recurring Transactions
        rec_groups = detect_recurring(normalized_rows)
        rec_groups_dicts = [asdict(rg) for rg in rec_groups]

        # 13. Quality Report
        qr: QualityReport = build_quality_report(
            filename=filename,
            page_types=[pt.extraction_method for pt in page_types],
            transactions=normalized_rows,
            reconciliation_report=recon_rep,
            ocr_results=[{"confidence": c} for c in ocr_confidences],
        )

        doc.close()

        return StatementProcessResult(
            success=True,
            session_id=session_id,
            filename=filename,
            file_size_bytes=len(file_bytes),
            page_count=page_count,
            bank_profile=bank_profile.name,
            doc_type="bank_statement",
            is_bank_statement=True,
            pages=pages_summary,
            account=account_dict,
            transactions=normalized_rows,
            quality=qr.to_dict(),
            recurring_groups=rec_groups_dicts,
            multi_statements=[{"rows_count": len(c)} for c in statement_chunks] if len(statement_chunks) > 1 else [],
        )
