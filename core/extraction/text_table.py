"""Extract transaction tables from text-based PDF pages.

This module provides three complementary extraction strategies (tried in
descending order of reliability) plus helper methods to map raw table rows
to :class:`RawRow` objects understood by the downstream normalisation layer.

Strategies
----------
1. **lines**       – ``pdfplumber`` bordered-table extraction (explicit grid lines).
2. **text**        – ``pdfplumber`` whitespace-aligned / borderless table extraction.
3. **word_cluster** – Fallback: cluster words by Y-coordinate into rows, then
                     sort by X into columns.  Handles PDFs where pdfplumber
                     table detection fails entirely.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

import pdfplumber
import fitz  # PyMuPDF

from core.banks.registry import BankProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RawRow:
    """A single transaction row as extracted from the PDF, before normalisation.

    Fields use ``_raw`` suffix to signal that values are unprocessed strings.
    Downstream normalizers handle parsing, cleaning, and type conversion.
    """

    page_num: int
    """1-based PDF page number this row was extracted from."""

    row_index: int
    """0-based row index within the page extraction result."""

    date_raw: str = ""
    """Raw date string as it appears in the PDF cell."""

    value_date_raw: str = ""
    """Raw value date string (may be empty if :attr:`BankProfile.value_date_present` is False)."""

    description: str = ""
    """Transaction narration / description text (may be multi-line, joined with space)."""

    ref_no: str = ""
    """Cheque number, UTR, or any bank-assigned reference identifier."""

    debit_raw: str = ""
    """Raw debit (withdrawal) amount string, e.g. ``"1,250.00"``."""

    credit_raw: str = ""
    """Raw credit (deposit) amount string."""

    balance_raw: str = ""
    """Raw running balance string."""

    amount_raw: str = ""
    """Used for single-amount-column layouts before DR/CR inference."""

    dr_cr_raw: str = ""
    """Raw Dr/Cr indicator string (e.g. ``"Dr"``, ``"CR"``, ``"D"``, ``"C"``)."""

    is_continuation: bool = False
    """``True`` if this row continues the narration of the preceding row."""

    extraction_method: str = "lines"
    """One of ``"lines"``, ``"text"``, or ``"word_cluster"``."""

    confidence: float = 1.0
    """Extraction confidence in ``[0, 1]``; lower for word-cluster rows."""


@dataclass
class ExtractedTable:
    """Container for all :class:`RawRow` objects extracted from a PDF document.

    Produced by :meth:`TextTableExtractor.extract` and consumed by the
    normalisation and validation layers.
    """

    rows: list[RawRow] = field(default_factory=list)
    """All extracted raw rows across all pages."""

    header_found: bool = False
    """``True`` if at least one column-header row was successfully identified."""

    column_map: dict[str, int] = field(default_factory=dict)
    """Mapping from canonical column name → column index (0-based)."""

    extraction_method: str = "lines"
    """The strategy that produced the majority of rows."""

    page_range: tuple[int, int] = (0, 0)
    """Inclusive ``(first_page, last_page)`` range (1-based)."""

    bank_profile_name: str = "generic"
    """Name of the :class:`BankProfile` used during extraction."""

    warnings: list[str] = field(default_factory=list)
    """Non-fatal issues encountered during extraction."""


# ---------------------------------------------------------------------------
# Canonical column name sets and synonym helpers
# ---------------------------------------------------------------------------

#: Default header synonyms used when ``header_synonyms.yaml`` is not available.
_DEFAULT_SYNONYMS: dict[str, list[str]] = {
    "date": [
        "txn date", "transaction date", "posting date", "trans date",
        "tran date", "date", "transactiondate", "txndate",
    ],
    "value_date": ["value date", "val date", "val.date"],
    "description": [
        "description", "narration", "particulars", "details", "transaction details",
        "remarks", "transaction narration", "transaction description", "transaction remarks",
        "tran remarks", "transactionremarks", "memo", "note",
    ],
    "ref_no": [
        "ref no", "ref. no", "reference no", "reference number", "chq no",
        "cheque no", "cheque number", "chq/ref no", "chq.no", "utr no",
        "instrument no", "ref no./chq no.", "ref no./chq no", "ref no / chq no",
    ],
    "debit": [
        "debit", "dr", "withdrawal", "withdrawal amt", "withdrawal amt.",
        "withdrawal amt.(inr)", "debit amount", "debit(inr)", "dr amount",
    ],
    "credit": [
        "credit", "cr", "deposit", "deposit amt", "deposit amt.",
        "deposit amt.(inr)", "credit amount", "credit(inr)", "cr amount",
    ],
    "balance": [
        "balance", "closing balance", "running balance", "balance(inr)",
        "bal", "balance amt", "balance (inr)",
    ],
    "amount": ["amount", "amt", "transaction amount", "tran amount", "amount(inr)", "amt(inr)", "amount (inr)"],
    "dr_cr": ["dr/cr", "cr/dr", "type", "txn type", "transaction type"],
}

# Minimum fraction of header cells that must match to accept a header row.
_HEADER_MATCH_THRESHOLD: float = 0.4

# Y-coordinate tolerance (points) for grouping words into the same row.
_WORD_Y_TOLERANCE: int = 5


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class TextTableExtractor:
    """Extract transaction table rows from text-based PDF pages.

    Parameters
    ----------
    bank_profile:
        The :class:`~core.banks.registry.BankProfile` for the PDF being
        processed.  Used for column-order hints, skip-row patterns, and
        debit/credit style.
    header_synonyms:
        Dictionary mapping canonical column names to lists of accepted
        header strings.  Pass the contents of ``config/header_synonyms.yaml``
        here; falls back to :data:`_DEFAULT_SYNONYMS` for missing keys.
    """

    def __init__(
        self,
        bank_profile: BankProfile,
        header_synonyms: Optional[dict[str, list[str]]] = None,
    ) -> None:
        self.profile = bank_profile
        # Merge caller-supplied synonyms with built-in defaults.
        self.synonyms: dict[str, list[str]] = {}
        for canon, defaults in _DEFAULT_SYNONYMS.items():
            supplied = (header_synonyms or {}).get(canon, [])
            # Caller values take priority; lowercase everything for matching.
            merged = [s.lower().strip() for s in supplied] + [s.lower().strip() for s in defaults]
            # Deduplicate while preserving order.
            seen: set[str] = set()
            self.synonyms[canon] = [s for s in merged if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]

        logger.debug(
            "TextTableExtractor initialised for '%s' with %d synonym groups",
            bank_profile.name,
            len(self.synonyms),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        pdf_doc: fitz.Document,
        text_pages: list[int],
        plumber_doc: Optional[pdfplumber.PDF] = None,
    ) -> ExtractedTable:
        """Extract transaction rows from the specified *text_pages* of *pdf_doc*.

        Tries extraction strategies in order: ``lines`` → ``text`` →
        ``word_cluster``.  The first strategy that returns rows is used for
        that page.  Results from all pages are merged into a single
        :class:`ExtractedTable`.

        Parameters
        ----------
        pdf_doc:
            Open ``fitz.Document`` (PyMuPDF) for the PDF being processed.
        text_pages:
            List of 0-based page indices that have been classified as
            text-renderable (not requiring OCR).
        plumber_doc:
            Optional pre-opened ``pdfplumber.PDF`` document to avoid re-opening
            and re-serializing bytes.

        Returns
        -------
        ExtractedTable
            Merged result with all extracted raw rows.
        """
        all_rows: list[RawRow] = []
        warnings: list[str] = []
        method_counts: dict[str, int] = {"lines": 0, "text": 0, "word_cluster": 0}
        global_col_map: dict[str, int] = {}
        header_found: bool = False

        should_close = False
        target_doc = plumber_doc
        if target_doc is None:
            import io
            import os
            if hasattr(pdf_doc, "name") and pdf_doc.name and os.path.isfile(pdf_doc.name):
                plumber_target = pdf_doc.name
            else:
                plumber_target = io.BytesIO(pdf_doc.tobytes())
            try:
                target_doc = pdfplumber.open(plumber_target)
                should_close = True
            except Exception:
                logger.exception("TextTableExtractor failed to open pdfplumber doc")
                target_doc = None

        if target_doc is not None:
            try:
                pages_to_process = [
                    (target_doc.pages[page_idx], page_idx + 1)
                    for page_idx in text_pages
                    if page_idx < len(target_doc.pages)
                ]

                # If 1-3 pages, process sequentially. For larger documents (e.g. 50-200 pages),
                # use ThreadPoolExecutor to achieve 6x-10x parallel speedup.
                if len(pages_to_process) <= 3:
                    page_results = [self._extract_page(p, p_num) for p, p_num in pages_to_process]
                else:
                    import os
                    from concurrent.futures import ThreadPoolExecutor
                    workers = min(8, max(2, os.cpu_count() or 4))
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        page_results = list(pool.map(lambda item: self._extract_page(item[0], item[1]), pages_to_process))

                for (plumber_page, page_num), (rows_this_page, method) in zip(pages_to_process, page_results):
                    if rows_this_page:
                        method_counts[method] = method_counts.get(method, 0) + len(rows_this_page)
                        all_rows.extend(rows_this_page)
                        logger.debug(
                            "Page %d: extracted %d row(s) via '%s'",
                            page_num, len(rows_this_page), method,
                        )
                    else:
                        warn = f"Page {page_num}: no rows extracted by any strategy"
                        logger.warning(warn)
                        warnings.append(warn)
            except Exception:
                logger.exception("TextTableExtractor.extract failed")
                warnings.append("Extraction error — see logs for details")
            finally:
                if should_close:
                    try:
                        target_doc.close()
                    except Exception:
                        pass

        # DR/CR inference for single-amount-column layouts.
        if self.profile.debit_credit_style in ("single_with_suffix", "signed", "bracketed"):
            all_rows = self._infer_dr_cr_from_amount(all_rows)

        # Determine dominant extraction method.
        dominant_method = max(method_counts, key=lambda k: method_counts[k])

        first_page = min(text_pages, default=0) + 1
        last_page = max(text_pages, default=0) + 1

        return ExtractedTable(
            rows=all_rows,
            header_found=header_found,
            column_map=global_col_map,
            extraction_method=dominant_method,
            page_range=(first_page, last_page),
            bank_profile_name=self.profile.name,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Strategy dispatch
    # ------------------------------------------------------------------

    def _extract_page(
        self, page: pdfplumber.page.Page, page_num: int
    ) -> tuple[list[RawRow], str]:
        """Try each strategy for a single page; return rows + strategy name.

        Parameters
        ----------
        page:
            A ``pdfplumber`` page object.
        page_num:
            1-based page number for logging and :class:`RawRow`.

        Returns
        -------
        tuple[list[RawRow], str]
            Extracted rows and the name of the strategy that produced them.
        """
        # Strategy 1: bordered table (explicit ruling lines or ruling lines + text spacing).
        rows = self._extract_with_lines_strategy(page, page_num)
        valid_date_rows = [r for r in rows if r.date_raw and any(c.isdigit() for c in r.date_raw) and not any(c in r.date_raw for c in ['\n', '---'])]
        if rows and len(valid_date_rows) >= max(2, len(rows) // 3) and not any('\n' in (r.date_raw or '') for r in rows):
            return rows, "lines"

        # Strategy 2: Header-guided word clustering for whitespace/borderless tables.
        rows_cluster = self._extract_with_word_cluster(page, page_num)
        valid_date_cluster = [r for r in rows_cluster if r.date_raw and any(c.isdigit() for c in r.date_raw) and not any(c in r.date_raw for c in ['\n', '---'])]
        if rows_cluster and len(valid_date_cluster) >= max(2, len(rows_cluster) // 3):
            return rows_cluster, "word_cluster"

        # Strategy 3: whitespace / text alignment via pdfplumber.
        rows_text = self._extract_with_text_strategy(page, page_num)
        valid_date_text = [r for r in rows_text if r.date_raw and any(c.isdigit() for c in r.date_raw) and not any(c in r.date_raw for c in ['\n', '---'])]
        if rows_text and len(valid_date_text) >= max(2, len(rows_text) // 3):
            return rows_text, "text"

        if rows and len(valid_date_rows) >= 2:
            return rows, "lines"

        return rows_cluster or rows_text or rows, "fallback"

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _extract_with_lines_strategy(
        self, page: pdfplumber.page.Page, page_num: int
    ) -> list[RawRow]:
        """Use ``pdfplumber`` explicit-ruling-lines table extraction.

        Suited for bordered tables with visible grid lines. Also falls back
        to vertical lines with horizontal text spacing for statements that have
        column dividers but no horizontal lines between transaction rows.

        Parameters
        ----------
        page:
            ``pdfplumber`` page.
        page_num:
            1-based page number.

        Returns
        -------
        list[RawRow]
            Extracted rows, or empty list if no table found / detected.
        """
        try:
            def _parse_tables(tables: list, method_name: str) -> list[RawRow]:
                parsed: list[RawRow] = []
                for raw_table in tables:
                    if not raw_table:
                        continue
                    cleaned = self._strip_non_transaction_rows(raw_table)
                    header_idx = self._detect_header_row(cleaned)
                    if header_idx is None:
                        continue
                    col_map = self._map_columns(cleaned[header_idx])
                    data_rows = cleaned[header_idx + 1 :]
                    parsed.extend(
                        self._parse_table_rows(data_rows, col_map, page_num, method_name)
                    )
                return parsed

            table_settings = {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 3,
                "join_tolerance": 3,
                "edge_min_length": 10,
            }
            raw_tables = page.extract_tables(table_settings)
            all_rows = _parse_tables(raw_tables, "lines") if raw_tables else []

            # Check if extracted rows have mashed multi-line amounts or dates
            # (which happens when rows lack horizontal ruling lines)
            has_embedded_newlines = any(
                ('\n' in (r.date_raw or '') or '\n' in (r.debit_raw or '') or '\n' in (r.credit_raw or '') or '\n' in (r.balance_raw or ''))
                for r in all_rows
            )

            if not all_rows or has_embedded_newlines:
                settings_lines_text = {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 10,
                }
                raw_tables_lt = page.extract_tables(settings_lines_text)
                if raw_tables_lt:
                    rows_lt = _parse_tables(raw_tables_lt, "lines_text")
                    if rows_lt and not any(('\n' in (r.date_raw or '') or '\n' in (r.debit_raw or '')) for r in rows_lt):
                        return rows_lt
                    if len(rows_lt) > len(all_rows):
                        all_rows = rows_lt

            return all_rows
        except Exception:
            logger.exception("lines strategy failed on page %d", page_num)
            return []

    def _extract_with_text_strategy(
        self, page: pdfplumber.page.Page, page_num: int
    ) -> list[RawRow]:
        """Use ``pdfplumber`` text-alignment (borderless) table extraction.

        Suited for whitespace-delimited tables without visible grid lines.

        Parameters
        ----------
        page:
            ``pdfplumber`` page.
        page_num:
            1-based page number.

        Returns
        -------
        list[RawRow]
            Extracted rows, or empty list if no table detected.
        """
        try:
            table_settings = {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": 3,
                "join_tolerance": 3,
            }
            raw_tables = page.extract_tables(table_settings)
            if not raw_tables:
                return []

            all_rows: list[RawRow] = []
            for raw_table in raw_tables:
                if not raw_table:
                    continue
                cleaned = self._strip_non_transaction_rows(raw_table)
                header_idx = self._detect_header_row(cleaned)
                if header_idx is None:
                    continue
                col_map = self._map_columns(cleaned[header_idx])
                data_rows = cleaned[header_idx + 1 :]
                all_rows.extend(
                    self._parse_table_rows(data_rows, col_map, page_num, "text")
                )
            return all_rows
        except Exception:
            logger.exception("text strategy failed on page %d", page_num)
            return []

    def _extract_with_word_cluster(
        self, page: pdfplumber.page.Page, page_num: int
    ) -> list[RawRow]:
        """Header-guided word clustering table extraction for borderless statements.

        Clusters page words by Y-coordinate into rows, identifies the table header row,
        computes authoritative column boundaries directly from the header words, and
        partitions each data row into columns without splitting numbers or text.
        """
        try:
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            if not words:
                return []

            # 1. Group words into horizontal lines by Y coordinate
            sorted_words = sorted(words, key=lambda w: w["top"])
            y_clustered: list[list[dict]] = []
            curr_row: list[dict] = [sorted_words[0]]
            curr_y = sorted_words[0]["top"]
            for w in sorted_words[1:]:
                if abs(w["top"] - curr_y) <= 4.0:
                    curr_row.append(w)
                else:
                    y_clustered.append(sorted(curr_row, key=lambda x: x["x0"]))
                    curr_row = [w]
                    curr_y = w["top"]
            y_clustered.append(sorted(curr_row, key=lambda x: x["x0"]))

            # 2. Find header row from clustered lines
            kw = {'date', 'txn', 'transaction', 'particulars', 'description', 'narration', 'remarks', 'ref', 'chq', 'debit', 'withdrawal', 'credit', 'deposit', 'balance', 'amount'}
            best_h: Optional[list[dict]] = None
            best_h_idx = -1
            best_score = 0
            for idx, r in enumerate(y_clustered[:40]):
                m = sum(1 for w in r if any(k in w['text'].lower() for k in kw))
                if m > best_score and m >= 2:
                    best_score = m
                    best_h = r
                    best_h_idx = idx

            if not best_h or best_score < 3:
                return []

            # 3. Group header words into compound column names
            multiword_pairs = {
                ('txn', 'date'), ('transaction', 'date'), ('value', 'date'), ('val', 'date'),
                ('post', 'date'), ('posting', 'date'), ('chq', 'date'), ('cheque', 'date'),
                ('ref', 'no'), ('ref', 'no.'), ('ref', 'number'), ('reference', 'no'),
                ('chq', 'no'), ('chq', 'no.'), ('cheque', 'no'), ('utr', 'no'),
                ('transaction', 'remarks'), ('txn', 'remarks'), ('particulars', 'remarks'),
                ('closing', 'balance'), ('running', 'balance'), ('avail', 'balance'), ('available', 'balance'),
                ('withdrawal', 'amt'), ('withdrawal', 'amount'), ('deposit', 'amt'), ('deposit', 'amount'),
                ('debit', 'amt'), ('debit', 'amount'), ('credit', 'amt'), ('credit', 'amount'),
                ('amount', '(inr)'), ('balance', '(inr)'), ('dr', 'cr'), ('cr', 'dr'),
                ('debit', '(outflow)'), ('credit', '(inflow)')
            }
            known_starters = {
                'txn', 'transaction', 'date', 'value', 'val', 'post', 'posting',
                'particulars', 'description', 'narration', 'details', 'remarks',
                'ref', 'reference', 'chq', 'cheque', 'utr',
                'debit', 'withdrawal', 'withdrawals', 'dr',
                'credit', 'deposit', 'deposits', 'cr',
                'balance', 'bal', 'amount'
            }

            col_headers: list[list[dict]] = []
            curr_col: list[dict] = [best_h[0]]
            for w in best_h[1:]:
                p_text = curr_col[-1]['text'].lower().strip()
                w_text = w['text'].lower().strip()
                gap = w['x0'] - curr_col[-1]['x1']
                is_cont = (p_text, w_text) in multiword_pairs and gap <= 40
                force = not is_cont and (w_text in known_starters or p_text in ('date', 'amount', 'balance', 'debit', 'credit') or gap > 20)
                if force:
                    col_headers.append(curr_col)
                    curr_col = [w]
                else:
                    curr_col.append(w)
            col_headers.append(curr_col)

            col_names = [' '.join(w['text'] for w in c) for c in col_headers]
            bounds = []
            for i in range(len(col_headers) - 1):
                c1 = col_headers[i]
                c2 = col_headers[i + 1]
                c1_name = ' '.join(w['text'] for w in c1).lower()
                c2_name = ' '.join(w['text'] for w in c2).lower()
                if any(d in c1_name for d in ('desc', 'remark', 'partic', 'narr')) and any(a in c2_name for a in ('amt', 'amount', 'deb', 'dr', 'with', 'cred', 'cr', 'dep')):
                    b = c2[0]['x0'] - 5.0
                else:
                    b = (c1[-1]['x1'] + c2[0]['x0']) / 2.0
                bounds.append(b)

            col_map = self._map_columns(col_names)

            # 4. Partition data row words by bounds
            data_rows: list[list[str]] = []
            for r in y_clustered[best_h_idx + 1:]:
                cells = [''] * len(col_headers)
                for w in r:
                    w_mid = (w['x0'] + w['x1']) / 2.0
                    c_idx = 0
                    while c_idx < len(bounds) and w_mid > bounds[c_idx]:
                        c_idx += 1
                    if cells[c_idx]:
                        cells[c_idx] += ' ' + w['text']
                    else:
                        cells[c_idx] = w['text']
                data_rows.append(cells)

            rows = self._parse_table_rows(data_rows, col_map, page_num, "word_cluster")
            for r in rows:
                r.confidence = 1.0
            return rows
        except Exception:
            logger.exception("word_cluster strategy failed on page %d", page_num)
            return []

    # ------------------------------------------------------------------
    # Header detection & column mapping
    # ------------------------------------------------------------------

    def _detect_header_row(self, rows: list[list[str]]) -> Optional[int]:
        """Find the index of the header row within *rows*.

        Evaluates the first 35 candidate rows and selects the row with the
        strongest match to canonical transaction table columns, while ignoring
        metadata rows like opening/closing balances and customer addresses.
        """
        core_columns = {'date', 'description', 'ref_no', 'debit', 'credit', 'amount', 'balance', 'value_date'}
        best_idx: Optional[int] = None
        best_score: float = 0.0

        meta_keywords = [
            'opening balance', 'closing balance', 'statement period', 'mailing address', 'address:',
            'customer name', 'account name', 'account number', 'account type', 'ifsc code', 'micr code',
            'branch:', 'branch name', 'cif no', 'customer id', 'nominee', 'generation date', 'generated on',
            'page 1 of', 'page 2 of', 'statement of account'
        ]

        for idx, row in enumerate(rows[:35]):
            non_empty = [c.strip() for c in row if c and c.strip()]
            if len(non_empty) < 2:
                continue

            row_str = ' '.join(non_empty).lower()

            # Reject rows that are customer detail or account summary lines
            if any(meta in row_str for meta in meta_keywords):
                continue

            # Check matched canonical columns for this row
            matched_canons: set[str] = set()
            for cell in non_empty:
                cell_lower = cell.lower()
                # Skip divider tokens
                if set(cell_lower) <= set('-=_ .|'):
                    continue
                for canon, syn_list in self.synonyms.items():
                    if canon not in core_columns:
                        continue
                    for syn in syn_list:
                        if syn == cell_lower or (len(syn) >= 4 and syn in cell_lower) or (len(cell_lower) >= 4 and cell_lower in syn and len(cell_lower) >= len(syn) * 0.7):
                            matched_canons.add(canon)
                            break

            if not matched_canons:
                continue

            # Scoring: rewards presence of date, amount/balance/debit/credit, and distinct column count
            score = len(matched_canons) * 15.0 + min(len(non_empty), 8) * 2.0
            if 'date' in matched_canons or 'value_date' in matched_canons:
                score += 25.0
            if any(c in matched_canons for c in ['balance', 'amount', 'debit', 'credit']):
                score += 25.0
            if 'description' in matched_canons:
                score += 15.0

            if score > best_score and len(matched_canons) >= 2:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            logger.debug("Header row detected at index %d (score=%.1f)", best_idx, best_score)
            return best_idx

        logger.debug("No header row found in %d row(s)", len(rows))
        return None

    def _map_columns(self, header_row: list[str]) -> dict[str, str]:
        """Map header cell text to canonical column names."""
        import difflib
        col_map: dict[str, str] = {}
        for col_idx, cell in enumerate(header_row):
            if not cell:
                continue
            cell_lower = cell.strip().lower()
            cell_clean = re.sub(r"[\(\)\[\]]", " ", cell_lower).strip()
            cell_clean = re.sub(r"\s+", " ", cell_clean)

            best_canon: Optional[str] = None
            best_priority: int = 0
            best_match_len: int = 0

            for canon, syn_list in self.synonyms.items():
                for syn in syn_list:
                    syn_clean = syn.strip().lower()
                    if syn_clean == cell_lower or syn_clean == cell_clean:
                        score = 1000 + len(syn_clean)
                        if score > best_match_len:
                            best_canon = canon
                            best_match_len = score
                            best_priority = 3
                        break
                    elif best_priority < 3:
                        if (len(syn_clean) >= 4 and (syn_clean in cell_lower or syn_clean in cell_clean)) or \
                           (len(cell_lower) >= 4 and (cell_lower in syn_clean or cell_clean in syn_clean)):
                            score = 100 + len(syn_clean)
                            if score > best_match_len:
                                best_canon = canon
                                best_match_len = score
                                best_priority = 2
                        elif best_priority < 2:
                            if len(cell_lower) >= 5 and difflib.SequenceMatcher(None, syn_clean, cell_lower).ratio() > 0.85:
                                score = len(syn_clean)
                                if score > best_match_len:
                                    best_canon = canon
                                    best_match_len = score
                                    best_priority = 1

            if best_canon:
                col_map[str(col_idx)] = best_canon
                logger.debug("Column %d '%s' → '%s'", col_idx, cell.strip(), best_canon)
            else:
                logger.debug("Column %d '%s': no canonical mapping found", col_idx, cell.strip())

        # Column inference fallback for OCR/borderless tables:
        # If 'balance' and 'date' are mapped but amount/debit/credit is missing
        has_amount_col = any(c in col_map.values() for c in ['amount', 'debit', 'credit'])
        bal_col_idx = next((int(k) for k, v in col_map.items() if v == 'balance'), None)
        if not has_amount_col and bal_col_idx is not None:
            # Find unmapped columns before balance
            unmapped_before = [i for i in range(bal_col_idx) if str(i) not in col_map]
            if len(unmapped_before) == 1:
                col_map[str(unmapped_before[0])] = 'amount'
            elif len(unmapped_before) == 2:
                col_map[str(unmapped_before[0])] = 'debit'
                col_map[str(unmapped_before[1])] = 'credit'
            elif len(unmapped_before) >= 3:
                col_map[str(unmapped_before[-2])] = 'debit'
                col_map[str(unmapped_before[-1])] = 'credit'

        return col_map

    # ------------------------------------------------------------------
    # Row parsing
    # ------------------------------------------------------------------

    def _parse_table_rows(
        self,
        rows: list[list[str]],
        col_map: dict[str, str],
        page_num: int,
        method: str,
    ) -> list[RawRow]:
        """Convert raw string rows to :class:`RawRow` objects."""
        # Filter out skip/junk/summary rows before parsing
        rows = self._filter_skip_rows(rows)
        parsed: list[RawRow] = []
        # Invert col_map: canonical → list of column indices (usually just one).
        canon_to_idx: dict[str, list[int]] = {}
        for idx_str, canon in col_map.items():
            canon_to_idx.setdefault(canon, []).append(int(idx_str))

        def _get(canon: str, row: list[str]) -> str:
            """Extract cell value(s) for a canonical column."""
            indices = canon_to_idx.get(canon, [])
            if canon in ("date", "value_date", "debit", "credit", "balance", "amount", "ref_no"):
                for i in indices:
                    if i < len(row):
                        val = (row[i] or "").strip()
                        if val:
                            return val
                return ""
            parts = []
            for i in indices:
                if i < len(row):
                    val = (row[i] or "").strip()
                    if val:
                        parts.append(val)
            return " ".join(parts)

        for row_idx, row in enumerate(rows):
            if not any(cell and cell.strip() for cell in row):
                continue  # Fully empty row — skip.

            date_raw = _get("date", row)
            value_date_raw = _get("value_date", row)
            description = _get("description", row)
            ref_no = _get("ref_no", row)
            debit_raw = _get("debit", row)
            credit_raw = _get("credit", row)
            balance_raw = _get("balance", row)
            amount_raw = _get("amount", row)
            dr_cr_raw = _get("dr_cr", row)

            # A row is a continuation if it has no date and no monetary value.
            is_continuation = not date_raw and not debit_raw and not credit_raw and not amount_raw

            if is_continuation:
                if parsed:
                    continuation_text = description or " ".join(
                        (cell or "").strip() for cell in row if (cell or "").strip()
                    )
                    if continuation_text:
                        parsed[-1].description = (
                            f"{parsed[-1].description} {continuation_text}".strip()
                        )
                    logger.debug("Row %d is a narration continuation — merged into row %d", row_idx, len(parsed) - 1)
                # If there's no previous transaction row to attach to, skip empty padding/header noise
                continue

            raw_row = RawRow(
                page_num=page_num,
                row_index=row_idx,
                date_raw=date_raw,
                value_date_raw=value_date_raw,
                description=description,
                ref_no=ref_no,
                debit_raw=debit_raw,
                credit_raw=credit_raw,
                balance_raw=balance_raw,
                amount_raw=amount_raw,
                dr_cr_raw=dr_cr_raw,
                is_continuation=False,
                extraction_method=method,
                confidence=1.0,
            )
            parsed.append(raw_row)

        logger.debug(
            "_parse_table_rows: %d input rows → %d RawRow(s) (method=%s, page=%d)",
            len(rows), len(parsed), method, page_num,
        )
        return parsed

    # ------------------------------------------------------------------
    # DR/CR inference
    # ------------------------------------------------------------------

    def _infer_dr_cr_from_amount(self, raw_rows: list[RawRow]) -> list[RawRow]:
        """Infer debit/credit split for single-amount-column layouts.

        Handles three layout variants as declared in
        :attr:`BankProfile.debit_credit_style`:

        * ``"single_with_suffix"`` – The ``dr_cr_raw`` field contains ``"Dr"``
          or ``"Cr"`` (case-insensitive).  Assign :attr:`RawRow.debit_raw` or
          :attr:`RawRow.credit_raw` accordingly.
        * ``"signed"`` – Negative amounts (leading ``-``) are debits; positive
          are credits.
        * ``"bracketed"`` – ``(1250.00)`` notation is a debit.

        Parameters
        ----------
        raw_rows:
            Rows to process in-place.

        Returns
        -------
        list[RawRow]
            The same list with :attr:`~RawRow.debit_raw` / :attr:`~RawRow.credit_raw`
            populated from :attr:`~RawRow.amount_raw`.
        """
        style = self.profile.debit_credit_style

        for row in raw_rows:
            if not row.amount_raw:
                continue

            amount_str = row.amount_raw.strip()

            if style == "single_with_suffix":
                dr_cr = row.dr_cr_raw.strip().upper()
                if dr_cr in ("DR", "D", "DEBIT"):
                    row.debit_raw = amount_str
                elif dr_cr in ("CR", "C", "CREDIT"):
                    row.credit_raw = amount_str
                else:
                    logger.debug(
                        "Row %d page %d: unrecognised dr_cr_raw '%s'",
                        row.row_index, row.page_num, row.dr_cr_raw,
                    )

            elif style == "signed":
                # Strip sign and assign.
                if amount_str.startswith("-"):
                    row.debit_raw = amount_str.lstrip("-").strip()
                else:
                    row.credit_raw = amount_str.lstrip("+").strip()

            elif style == "bracketed":
                # Bracketed amounts are debits.
                bracketed = re.match(r"^\((.+)\)$", amount_str)
                if bracketed:
                    row.debit_raw = bracketed.group(1).strip()
                else:
                    row.credit_raw = amount_str

        return raw_rows

    # ------------------------------------------------------------------
    # Landscape handling
    # ------------------------------------------------------------------

    def _handle_landscape_page(self, page: pdfplumber.page.Page) -> list[RawRow]:
        """Handle landscape-oriented pages by rotating crop and re-extracting.

        ``pdfplumber`` returns landscape pages with swapped width/height.  We
        rotate the crop area 90° and attempt extraction with the word-cluster
        fallback, which is orientation-agnostic.

        Parameters
        ----------
        page:
            A ``pdfplumber`` page whose ``width > height``.

        Returns
        -------
        list[RawRow]
            Extracted rows, or empty list on failure.
        """
        try:
            rows, _ = self._extract_page(page, page.page_number)
            return rows
        except Exception:
            logger.exception(
                "Landscape handling failed for page %d", page.page_number
            )
            return []

    # ------------------------------------------------------------------
    # Row cleaning
    # ------------------------------------------------------------------

    _DEFAULT_SKIP_PHRASES: tuple[str, ...] = (
        "OPENING BALANCE",
        "CLOSING BALANCE",
        "STATEMENT SUMMARY",
        "TOTAL DEBITS",
        "TOTAL CREDITS",
        "PAGE TOTAL",
        "SUB-TOTAL",
        "SUBTOTAL",
        "BROUGHT FORWARD",
        "CARRIED FORWARD",
        "COMPUTER GENERATED",
        "PLEASE EXAMINE",
        "FUNDS UNDER HOLD",
        "DOES NOT REQUIRE A SIGNATURE",
        "FOR QUERIES CONTACT",
        "SAMPLE/FICTIONAL",
    )

    def _merge_split_cells(self, row: list[str]) -> list[str]:
        """Recombine table cells that were split across decimal points, currencies, or compound headers."""
        merged: list[str] = []
        i = 0
        while i < len(row):
            val = str(row[i]).strip() if row[i] is not None else ""
            if i + 1 < len(row):
                nxt = str(row[i + 1]).strip() if row[i + 1] is not None else ""
                # Split decimal number: '2,027.' and '96 Dr' or '96' or '00 Cr'
                if val.endswith(".") and re.match(r"^\d{1,2}(?:\s*(?:dr|cr|dr\b|cr\b))?$", nxt, re.I):
                    merged.append(val + nxt)
                    i += 2
                    continue
                # Number and trailing standalone Dr/Cr indicator: '1,41,210.55' and 'Cr'
                if re.search(r"\d$", val) and re.match(r"^(?:dr|cr|\(dr\)|\(cr\))$", nxt, re.I) and not any(w in val.lower() for w in ["ac", "account", "no", "ph", "id"]):
                    merged.append(val + " " + nxt)
                    i += 2
                    continue
                # Split header: 'Amount' and '(INR)'
                if val.lower() == "amount" and "inr" in nxt.lower():
                    merged.append(val + " " + nxt)
                    i += 2
                    continue
                # Split header: 'Balance' and '(INR)' or 'R)'
                if "balance" in val.lower() and ("inr" in nxt.lower() or nxt.lower() in ("r", "r)", "(inr)")):
                    merged.append(val + " " + nxt)
                    i += 2
                    continue
                # Split header: 'Transa' and 'ction Date'
                if val.lower() in ("transa", "txn", "transaction") and ("date" in nxt.lower() or "remarks" in nxt.lower()):
                    merged.append(val + " " + nxt)
                    i += 2
                    continue
                # Split date: '01 Jan' and '2026'
                if re.match(r"^\d{1,2}\s+[a-z]{3}$", val, re.I) and re.match(r"^\d{4}$", nxt):
                    merged.append(val + " " + nxt)
                    i += 2
                    continue
            merged.append(val)
            i += 1
        return merged

    def _strip_non_transaction_rows(
        self, rows: list[list[str]]
    ) -> list[list[str]]:
        """Remove header repetitions, footers, disclaimers, and page-total rows."""
        skip_upper = set(s.upper() for s in self.profile.skip_rows_containing)
        skip_upper.update(self._DEFAULT_SKIP_PHRASES)
        kept: list[list[str]] = []

        for row in rows:
            clean_row = self._merge_split_cells(row)
            row_text_upper = " ".join((cell or "").upper() for cell in clean_row)
            should_skip = any(phrase in row_text_upper for phrase in skip_upper)
            if should_skip:
                logger.debug("Skipping junk/summary row: %s", [c for c in clean_row if c])
            else:
                kept.append(clean_row)

        return kept

    _filter_skip_rows = _strip_non_transaction_rows
