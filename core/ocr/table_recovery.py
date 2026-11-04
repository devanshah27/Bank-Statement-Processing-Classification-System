"""Recover table structure from OCR word bounding boxes.

Bank statement transactions are always tabular: date | description | debit |
credit | balance.  This module reconstructs that grid from the unordered word
bounding boxes emitted by the OCR engine.

Typical usage::

    rows = recover_table(ocr_result, page_width=2480)
    for row in rows:
        date, description, debit, credit, balance = row[:5]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .engine_base import OCRResult, OCRWord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OCRTableRow:
    """A single recovered table row.

    Attributes:
        words: All :class:`~engine_base.OCRWord` objects that belong to this row,
               sorted left-to-right.
        y_center: Median vertical centre of the row's words (pixels).
        cells: Text strings assigned to each detected column, in left-to-right
               order.  Empty string for empty cells.
    """

    words: list[OCRWord]
    y_center: float
    cells: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Word → row clustering
# ---------------------------------------------------------------------------


def cluster_words_into_rows(
    words: list[OCRWord],
    y_tolerance: int = 8,
) -> list[list[OCRWord]]:
    """Group OCR words into horizontal rows by proximity.

    Words are grouped using greedy 1-D clustering on the y-centre coordinate.
    Within each row, words are sorted left-to-right by their left edge.

    Args:
        words: List of :class:`~engine_base.OCRWord` objects from a single page.
        y_tolerance: Maximum gap between consecutive y-centres (pixels) that
                     still belongs to the same row.  Increase for large fonts;
                     decrease for dense tables.

    Returns:
        List of rows, where each row is itself a list of words sorted by
        x-position (left to right).
    """
    if not words:
        return []

    # Sort all words by vertical centre
    sorted_words = sorted(words, key=lambda w: w.y_center)

    rows: list[list[OCRWord]] = []
    current_row: list[OCRWord] = [sorted_words[0]]
    current_y = sorted_words[0].y_center

    for word in sorted_words[1:]:
        if abs(word.y_center - current_y) <= y_tolerance:
            current_row.append(word)
        else:
            rows.append(sorted(current_row, key=lambda w: w.left))
            current_row = [word]
            current_y = word.y_center

    rows.append(sorted(current_row, key=lambda w: w.left))

    logger.debug("Clustered %d words into %d rows", len(words), len(rows))
    return rows


# ---------------------------------------------------------------------------
# Column boundary detection
# ---------------------------------------------------------------------------


HEADER_KEYWORDS = {
    "date", "txn", "value", "particulars", "description", "narration",
    "details", "ref", "chq", "cheque", "debit", "withdrawal", "dr",
    "credit", "deposit", "cr", "balance", "bal", "amount"
}


def detect_column_boundaries(
    rows: list[list[OCRWord]],
    min_rows: int = 3,
) -> list[float]:
    """Detect x-positions of column separators from inter-word gaps.

    First checks for an authoritative table header row containing column labels
    (e.g., Date, Description, Debit, Credit, Balance). When found, boundaries
    are derived directly from the header words to avoid misclustering sparse
    columns (such as separate Debit/Credit columns).

    Falls back to multi-row gap clustering when no explicit header row is found.

    Args:
        rows: Output of :func:`cluster_words_into_rows`.
        min_rows: Minimum number of rows in which a gap must appear to be
                  considered a true column separator (not incidental spacing).

    Returns:
        Sorted list of x-coordinates marking column boundaries.
    """
    if not rows:
        return []

    # 1. Header-guided boundary detection: find candidate header row
    best_header_row: Optional[list[OCRWord]] = None
    max_matched_keywords = 0

    for row in rows:
        row_text_lower = " ".join(w.text.lower() for w in row)
        has_dr_cr = (
            ("debit" in row_text_lower or "withdrawal" in row_text_lower or "dr" in row_text_lower) and
            ("credit" in row_text_lower or "deposit" in row_text_lower or "cr" in row_text_lower)
        )
        matched_kw = sum(1 for w in row if any(k in w.text.lower() for k in HEADER_KEYWORDS))
        if (has_dr_cr and matched_kw >= 2) or matched_kw >= 4:
            if matched_kw > max_matched_keywords:
                max_matched_keywords = matched_kw
                best_header_row = row

    if best_header_row and len(best_header_row) >= 2:
        # Group adjacent words in the header row that belong to the same header phrase
        # (e.g. "Txn Date", "Value Date", "Ref No./Chq No.")
        multiword_pairs = {
            ("txn", "date"), ("transaction", "date"), ("value", "date"), ("val", "date"),
            ("post", "date"), ("posting", "date"), ("chq", "date"), ("cheque", "date"),
            ("ref", "no"), ("ref", "no."), ("ref", "number"), ("reference", "no"),
            ("chq", "no"), ("chq", "no."), ("cheque", "no"), ("utr", "no"),
            ("no./chq", "no"), ("no./chq", "no."), ("ref", "no./chq"),
            ("closing", "balance"), ("running", "balance"), ("avail", "balance"), ("available", "balance"),
            ("withdrawal", "amt"), ("withdrawal", "amount"), ("deposit", "amt"), ("deposit", "amount"),
            ("debit", "amt"), ("debit", "amount"), ("credit", "amt"), ("credit", "amount"),
            ("dr", "amount"), ("cr", "amount"), ("dr", "cr"), ("cr", "dr"),
            ("debit", "(outflow)"), ("credit", "(inflow)")
        }
        known_starters = {
            "txn", "transaction", "date", "value", "val", "post", "posting",
            "particulars", "description", "narration", "details", "remarks",
            "ref", "reference", "chq", "cheque", "utr",
            "debit", "withdrawal", "withdrawals", "dr",
            "credit", "deposit", "deposits", "cr",
            "balance", "bal"
        }

        col_headers: list[list[OCRWord]] = []
        curr_col: list[OCRWord] = [best_header_row[0]]
        for w in best_header_row[1:]:
            prev = curr_col[-1]
            p_text = prev.text.lower().strip()
            w_text = w.text.lower().strip()
            gap = w.left - prev.right

            is_continuation = (p_text, w_text) in multiword_pairs and gap <= 40
            force_split = (
                not is_continuation and (
                    w_text in known_starters or
                    p_text in ("date", "no.", "no", "description", "particulars", "narration", "debit", "credit", "balance", "amount") or
                    gap > 20
                )
            )
            if force_split:
                col_headers.append(curr_col)
                curr_col = [w]
            else:
                curr_col.append(w)
        col_headers.append(curr_col)

        if len(col_headers) >= 3:
            header_boundaries: list[float] = []
            for i in range(len(col_headers) - 1):
                c1 = col_headers[i]
                c2 = col_headers[i + 1]
                header_boundaries.append(float((c1[-1].right + c2[0].left) / 2.0))
            header_boundaries.sort()
            logger.info("Authoritative header boundaries detected: %s", header_boundaries)
            return header_boundaries

    # 2. Fallback: Multi-row gap histogram clustering

    # Collect the midpoint x-position of every inter-word gap
    gap_midpoints: list[float] = []
    for row in rows:
        if len(row) < 2:
            continue
        for i in range(len(row) - 1):
            right_of_left = row[i].right
            left_of_right = row[i + 1].left
            gap_width = left_of_right - right_of_left
            if gap_width > 5:  # ignore sub-pixel 'gaps'
                midpoint = (right_of_left + left_of_right) / 2.0
                gap_midpoints.append(midpoint)

    if len(gap_midpoints) < min_rows:
        logger.debug("Not enough gap midpoints (%d) to detect columns", len(gap_midpoints))
        return []

    # Merge nearby gap positions using a simple scan-line algorithm
    gap_midpoints.sort()
    merge_threshold = 20.0  # pixels
    clusters: list[list[float]] = []
    current_cluster: list[float] = [gap_midpoints[0]]

    for gx in gap_midpoints[1:]:
        if gx - current_cluster[-1] <= merge_threshold:
            current_cluster.append(gx)
        else:
            clusters.append(current_cluster)
            current_cluster = [gx]
    clusters.append(current_cluster)

    # Keep only clusters that appear in at least min_rows rows
    boundaries: list[float] = []
    for cluster in clusters:
        if len(cluster) >= min_rows:
            boundaries.append(float(np.median(cluster)))

    boundaries.sort()
    logger.debug("Detected %d column boundaries: %s", len(boundaries), boundaries)
    return boundaries


# ---------------------------------------------------------------------------
# Word → cell assignment
# ---------------------------------------------------------------------------


def assign_words_to_cells(
    row: list[OCRWord],
    column_boundaries: list[float],
) -> list[str]:
    """Assign words in a row to their corresponding table cells.

    Each word is placed in the cell whose boundary contains its x-centre.
    Words whose x-centre exceeds all boundaries are placed in the last cell.

    Args:
        row: Left-to-right sorted list of :class:`~engine_base.OCRWord` objects
             for a single table row.
        column_boundaries: Sorted list of x-coordinates from
                           :func:`detect_column_boundaries`.

    Returns:
        List of cell strings with length ``len(column_boundaries) + 1``.
        Each entry contains the space-joined text of all words in that cell.
    """
    num_cells = len(column_boundaries) + 1
    cells: list[list[str]] = [[] for _ in range(num_cells)]

    for word in row:
        col_index = num_cells - 1  # default: last column
        for i, boundary in enumerate(column_boundaries):
            if word.x_center <= boundary:
                col_index = i
                break
        cells[col_index].append(word.text)

    return [" ".join(tokens) for tokens in cells]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def recover_table(
    ocr_result: OCRResult,
    page_width: int,
) -> list[list[str]]:
    """Recover a full tabular structure from an :class:`~engine_base.OCRResult`.

    Runs the complete pipeline:
    words → row clustering → column detection → cell assignment.

    Args:
        ocr_result: The :class:`~engine_base.OCRResult` from the OCR engine for
                    a single page.
        page_width: Width of the source page in pixels (used for context /
                    future normalisation; not currently used in computation).

    Returns:
        List of rows, where each row is a list of cell strings.  Empty cells
        are represented as empty strings.  Returns an empty list if fewer than
        3 words were recognized.
    """
    words = ocr_result.words
    if not words:
        logger.warning("recover_table called with zero words")
        return []

    logger.info(
        "Recovering table structure from %d words (page width=%d px)",
        len(words), page_width,
    )

    # Step 1: Cluster words into horizontal rows
    raw_rows = cluster_words_into_rows(words)

    # Step 2: Detect column boundaries from inter-word gap analysis
    column_boundaries = detect_column_boundaries(raw_rows)

    if not column_boundaries:
        # Fallback: return one cell per row (no column structure found)
        logger.info("No column structure detected — returning single-column rows")
        return [[" ".join(w.text for w in row)] for row in raw_rows]

    # Step 3: Assign words to cells
    table: list[list[str]] = []
    for row in raw_rows:
        cells = assign_words_to_cells(row, column_boundaries)
        table.append(cells)

    logger.info(
        "Table recovered: %d rows × %d columns",
        len(table),
        len(column_boundaries) + 1,
    )
    return table
