"""Main table extraction orchestrator.

Routes each PDF page to the appropriate extraction path (text-based or OCR)
based on pre-computed :class:`PageTypeResult` classifications, then
normalises all raw rows into typed :class:`TransactionRow` objects.

This module is the primary entry point used by the pipeline layer.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import fitz  # PyMuPDF

from core.banks.registry import BankProfile
from core.extraction.text_table import ExtractedTable, RawRow, TextTableExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supporting type definitions (lightweight protocols / stubs)
# ---------------------------------------------------------------------------


@dataclass
class PageTypeResult:
    """Classification result for a single PDF page.

    Produced by the page-classifier stage and consumed here to determine
    whether a page should be processed via text extraction or OCR.
    """

    page_index: int
    """0-based page index within the PDF."""

    is_text: bool
    """``True`` if the page has sufficient embedded text for direct extraction."""

    is_scanned: bool
    """``True`` if the page appears to be a scanned image (needs OCR)."""

    text_char_count: int = 0
    """Number of text characters detected on the page (pre-OCR)."""

    image_area_fraction: float = 0.0
    """Fraction of page area covered by image objects."""


@dataclass
class ChunkContext:
    """Optional context for multi-chunk (streamed) processing.

    When a large PDF is split into chunks for memory-efficient processing,
    this carries cross-chunk state such as running year hints or carried-
    forward balances.
    """

    year_hint: int = 0
    """Calendar year inferred from an earlier chunk or document metadata."""

    carried_balance: Optional[float] = None
    """Running balance carried forward from the last row of the previous chunk."""

    chunk_index: int = 0
    """0-based index of the current chunk within the full document."""


# ---------------------------------------------------------------------------
# Output data class
# ---------------------------------------------------------------------------


@dataclass
class TransactionRow:
    """A fully normalised transaction row ready for classification and storage.

    All monetary fields use Python ``float``.  ``None`` indicates that the
    value could not be parsed from the source PDF.
    """

    row_id: str
    """UUID4 string uniquely identifying this row within a processing run."""

    page_num: int
    """1-based PDF page number this row was extracted from."""

    date: Optional[str] = None
    """ISO 8601 date string (``YYYY-MM-DD``), or ``None`` if parsing failed."""

    value_date: Optional[str] = None
    """ISO 8601 value date, or ``None``."""

    description: str = ""
    """Full transaction narration / description (multi-line parts joined)."""

    ref_no: str = ""
    """Cheque number, UTR, UPI reference, or other bank-assigned identifier."""

    debit: Optional[float] = None
    """Debit (withdrawal) amount in INR, or ``None``."""

    credit: Optional[float] = None
    """Credit (deposit) amount in INR, or ``None``."""

    balance: Optional[float] = None
    """Running account balance after this transaction, or ``None``."""

    extraction_method: str = "text"
    """The extraction path used: ``"text"``, ``"ocr_tesseract"``, ``"ocr_rapid"``, or ``"ocr_remote"``."""

    ocr_confidence: Optional[float] = None
    """Mean OCR word confidence ``[0, 1]`` for OCR-extracted rows; ``None`` for text rows."""

    is_flagged: bool = False
    """``True`` if this row has one or more quality / reconciliation issues."""

    flag_reasons: list[str] = field(default_factory=list)
    """Human-readable descriptions of each quality flag."""


# ---------------------------------------------------------------------------
# Normaliser protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RowNormalizer(Protocol):
    """Interface that all normaliser objects must implement."""

    def normalize(self, raw: RawRow, year_hint: int) -> dict[str, Any]:
        """Return a dict of normalised field values for a :class:`TransactionRow`.

        Parameters
        ----------
        raw:
            The raw row produced by an extraction strategy.
        year_hint:
            Calendar year to use when the date string omits the year.

        Returns
        -------
        dict[str, Any]
            Keys correspond to :class:`TransactionRow` field names.
        """
        ...


# ---------------------------------------------------------------------------
# OCR engine protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class OcrEngine(Protocol):
    """Interface that all OCR engine adapters must implement."""

    def extract_rows(self, page: fitz.Page) -> tuple[list[RawRow], float]:
        """Extract raw rows from a scanned page image.

        Parameters
        ----------
        page:
            A ``fitz.Page`` object from the open ``fitz.Document``.

        Returns
        -------
        tuple[list[RawRow], float]
            Extracted rows and mean OCR confidence ``[0, 1]``.
        """
        ...


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class TableExtractor:
    """Orchestrates table extraction across all pages of a PDF document.

    Routing logic:

    * Pages classified as **text** (``PageTypeResult.is_text=True``) are
      processed by :class:`~core.extraction.text_table.TextTableExtractor`.
    * Pages classified as **scanned** (``PageTypeResult.is_scanned=True``) are
      delegated to the injected *ocr_engine*.
    * Pages that are neither are skipped with a warning.

    After extraction, all :class:`RawRow` objects from both paths are passed
    through the *normalizers* chain to produce final :class:`TransactionRow`
    objects.

    Parameters
    ----------
    bank_profile:
        The detected / selected :class:`~core.banks.registry.BankProfile`.
    ocr_engine:
        An object implementing :class:`OcrEngine`; used for scanned pages.
        Pass ``None`` to skip OCR pages with a warning.
    text_extractor:
        A configured :class:`~core.extraction.text_table.TextTableExtractor`
        instance.
    normalizers:
        Ordered list of :class:`RowNormalizer` objects applied sequentially to
        each :class:`RawRow`.
    """

    def __init__(
        self,
        bank_profile: BankProfile,
        ocr_engine: Optional[OcrEngine],
        text_extractor: TextTableExtractor,
        normalizers: list[RowNormalizer],
    ) -> None:
        self.profile = bank_profile
        self.ocr_engine = ocr_engine
        self.text_extractor = text_extractor
        self.normalizers = normalizers

        logger.info(
            "TableExtractor initialised: bank='%s', ocr=%s, normalizers=%d",
            bank_profile.name,
            type(ocr_engine).__name__ if ocr_engine else "None",
            len(normalizers),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_all_pages(
        self,
        pdf_doc: fitz.Document,
        pdfplumber_doc: Any,
        page_types: list[PageTypeResult],
        chunk_context: Optional[ChunkContext] = None,
    ) -> list[TransactionRow]:
        """Extract all transactions from all pages, routing text vs OCR per page.

        Parameters
        ----------
        pdf_doc:
            Open ``fitz.Document`` for the PDF.
        pdfplumber_doc:
            Open ``pdfplumber`` document (passed to text extractor internally).
            Can be ``None`` if all pages are OCR-only.
        page_types:
            Per-page classification results, one per PDF page.
        chunk_context:
            Optional cross-chunk state.  When provided, its ``year_hint`` is
            used as the default year for date parsing.

        Returns
        -------
        list[TransactionRow]
            Fully normalised transaction rows sorted by ``(page_num, row_id)``.
        """
        year_hint = chunk_context.year_hint if chunk_context else 0
        all_raw_rows: list[RawRow] = []

        text_page_indices: list[int] = []
        ocr_page_indices: list[int] = []
        skipped_pages: list[int] = []

        for pt in page_types:
            if pt.is_text:
                text_page_indices.append(pt.page_index)
            elif pt.is_scanned:
                ocr_page_indices.append(pt.page_index)
            else:
                skipped_pages.append(pt.page_index)
                logger.debug(
                    "Page %d skipped: neither text nor scanned", pt.page_index + 1
                )

        logger.info(
            "extract_all_pages: %d text pages, %d OCR pages, %d skipped",
            len(text_page_indices),
            len(ocr_page_indices),
            len(skipped_pages),
        )

        # --- Text pages ---
        if text_page_indices:
            text_rows = self._process_text_pages(pdf_doc, text_page_indices)
            all_raw_rows.extend(text_rows)

        # --- OCR pages ---
        for page_idx in ocr_page_indices:
            page_num = page_idx + 1
            ocr_rows = self._process_ocr_page(pdf_doc[page_idx], page_num)
            all_raw_rows.extend(ocr_rows)

        logger.info(
            "Total raw rows collected: %d (text) + OCR pages processed individually",
            len(all_raw_rows),
        )

        # --- Normalise ---
        transaction_rows = self._normalize_rows(all_raw_rows, year_hint)
        logger.info("Normalisation complete: %d TransactionRow(s)", len(transaction_rows))

        return transaction_rows

    # ------------------------------------------------------------------
    # Page processing
    # ------------------------------------------------------------------

    def _process_text_pages(
        self, pdf_doc: fitz.Document, page_indices: list[int]
    ) -> list[RawRow]:
        """Delegate text pages to the :class:`TextTableExtractor`.

        Parameters
        ----------
        pdf_doc:
            Open ``fitz.Document``.
        page_indices:
            0-based page indices to extract.

        Returns
        -------
        list[RawRow]
            All raw rows from all text pages.
        """
        try:
            extracted: ExtractedTable = self.text_extractor.extract(
                pdf_doc, page_indices
            )
            logger.info(
                "_process_text_pages: strategy='%s', rows=%d, warnings=%d",
                extracted.extraction_method,
                len(extracted.rows),
                len(extracted.warnings),
            )
            for warning in extracted.warnings:
                logger.warning("TextExtractor warning: %s", warning)
            return extracted.rows
        except Exception:
            logger.exception("_process_text_pages failed")
            return []

    def _process_ocr_page(
        self, page: fitz.Page, page_num: int
    ) -> list[RawRow]:
        """Delegate a single scanned page to the OCR engine.

        Parameters
        ----------
        page:
            A ``fitz.Page`` from the open ``fitz.Document``.
        page_num:
            1-based page number for logging and :class:`RawRow`.

        Returns
        -------
        list[RawRow]
            Raw rows with ``extraction_method`` set to the OCR engine name and
            ``confidence`` set to the mean OCR confidence.
        """
        if self.ocr_engine is None:
            logger.warning(
                "Page %d is scanned but no OCR engine configured — skipping", page_num
            )
            return []

        try:
            rows, mean_confidence = self.ocr_engine.extract_rows(page)
            engine_name = type(self.ocr_engine).__name__.lower()
            # Determine method label from engine class name heuristic.
            if "tesseract" in engine_name:
                method = "ocr_tesseract"
            elif "rapid" in engine_name:
                method = "ocr_rapid"
            elif "remote" in engine_name:
                method = "ocr_remote"
            else:
                method = "ocr_unknown"

            for row in rows:
                row.page_num = page_num
                row.extraction_method = method
                row.confidence = mean_confidence

            logger.info(
                "_process_ocr_page: page=%d method='%s' rows=%d confidence=%.2f",
                page_num, method, len(rows), mean_confidence,
            )
            return rows
        except Exception:
            logger.exception("_process_ocr_page failed for page %d", page_num)
            return []

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _normalize_rows(
        self, raw_rows: list[RawRow], year_hint: int
    ) -> list[TransactionRow]:
        """Apply the normaliser chain to every raw row.

        Each :class:`RowNormalizer` in :attr:`normalizers` is applied in
        sequence.  Later normalisers can override earlier ones by overwriting
        dict keys.  Rows that produce no parseable date **and** no parseable
        monetary amounts are flagged but still included so the user can review
        them.

        Parameters
        ----------
        raw_rows:
            Raw rows from text extraction and/or OCR.
        year_hint:
            Calendar year used as fallback when date strings omit the year.

        Returns
        -------
        list[TransactionRow]
            Normalised rows.  Order matches the input order.
        """
        result: list[TransactionRow] = []

        for raw in raw_rows:
            # Accumulate field updates from each normaliser.
            merged: dict[str, Any] = {}
            for normalizer in self.normalizers:
                try:
                    updates = normalizer.normalize(raw, year_hint)
                    merged.update(updates)
                except Exception:
                    logger.exception(
                        "Normaliser %s failed on row %d page %d",
                        type(normalizer).__name__,
                        raw.row_index,
                        raw.page_num,
                    )

            # Determine extraction_method and ocr_confidence from the raw row.
            extracted_method = raw.extraction_method
            is_ocr = extracted_method.startswith("ocr")
            ocr_confidence: Optional[float] = raw.confidence if is_ocr else None

            # Build the TransactionRow.
            flag_reasons: list[str] = merged.pop("flag_reasons", [])
            is_flagged = bool(flag_reasons)

            txn = TransactionRow(
                row_id=str(uuid.uuid4()),
                page_num=raw.page_num,
                date=merged.get("date"),
                value_date=merged.get("value_date"),
                description=merged.get("description", raw.description),
                ref_no=merged.get("ref_no", raw.ref_no),
                debit=merged.get("debit"),
                credit=merged.get("credit"),
                balance=merged.get("balance"),
                extraction_method=extracted_method,
                ocr_confidence=ocr_confidence,
                is_flagged=is_flagged,
                flag_reasons=flag_reasons,
            )

            # Auto-flag rows with no date and no monetary values.
            if txn.date is None and txn.debit is None and txn.credit is None:
                if not txn.is_flagged:
                    txn.is_flagged = True
                txn.flag_reasons.append("no_parseable_date_or_amount")

            result.append(txn)

        return result
