"""Per-page type detection module (Native Text vs Scanned/OCR).

Determines whether a PDF page contains extractable native vector text or if it
is a scanned image requiring optical character recognition (OCR).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Any
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

DEFAULT_CHAR_THRESHOLD = 80
DEFAULT_WORD_THRESHOLD = 15
DEFAULT_IMAGE_AREA_THRESHOLD = 0.75  # 75% of page area covered by an image


@dataclass
class PageTypeResult:
    """Detection result for a single PDF page."""
    page_num: int
    extraction_method: str  # "text" | "ocr"
    char_count: int
    word_count: int
    has_large_image: bool
    confidence: float
    reason: str

    @property
    def is_text(self) -> bool:
        return self.extraction_method == "text"

    @property
    def is_scanned(self) -> bool:
        return self.extraction_method == "ocr"

    @property
    def page_index(self) -> int:
        return max(0, self.page_num - 1)


def detect_page_type(
    page: fitz.Page,
    page_num: int = 1,
    char_threshold: int = DEFAULT_CHAR_THRESHOLD,
    word_threshold: int = DEFAULT_WORD_THRESHOLD,
    image_area_threshold: float = DEFAULT_IMAGE_AREA_THRESHOLD,
) -> PageTypeResult:
    """Analyze a single PyMuPDF page to determine if it should be processed via native text or OCR.

    Args:
        page: PyMuPDF Page object.
        page_num: 1-indexed page number.
        char_threshold: Minimum character count to consider text layer viable.
        word_threshold: Minimum word count.
        image_area_threshold: Fraction of page covered by a single image indicating a scanned page.

    Returns:
        PageTypeResult specifying extraction_method as 'text' or 'ocr'.
    """
    rect = page.rect
    page_area = max(1.0, rect.width * rect.height)

    # 1. Inspect native text
    text = page.get_text("text").strip()
    char_count = len(text)
    words = text.split()
    word_count = len(words)

    # 2. Inspect embedded images
    image_list = page.get_images(full=True)
    has_large_image = False
    max_image_area_fraction = 0.0

    for img_info in image_list:
        xref = img_info[0]
        try:
            # Check bounding boxes where image is placed on page
            for rect_img in page.get_image_rects(xref):
                img_area = rect_img.width * rect_img.height
                fraction = img_area / page_area
                if fraction > max_image_area_fraction:
                    max_image_area_fraction = fraction
        except Exception as e:
            logger.debug("Could not determine image rects for xref %s: %s", xref, e)

    if max_image_area_fraction >= image_area_threshold:
        has_large_image = True

    # 3. Decision rule
    # If text is very scarce, or if a single image dominates the page and text is low:
    if char_count < char_threshold or word_count < word_threshold:
        return PageTypeResult(
            page_num=page_num,
            extraction_method="ocr",
            char_count=char_count,
            word_count=word_count,
            has_large_image=has_large_image,
            confidence=0.95 if has_large_image or char_count < 20 else 0.85,
            reason=f"Low text content ({char_count} chars, {word_count} words). Routed to OCR."
        )

    if has_large_image and char_count < 200:
        return PageTypeResult(
            page_num=page_num,
            extraction_method="ocr",
            char_count=char_count,
            word_count=word_count,
            has_large_image=True,
            confidence=0.90,
            reason=f"Page covered by full-page scan ({max_image_area_fraction:.1%} area). Routed to OCR."
        )

    return PageTypeResult(
        page_num=page_num,
        extraction_method="text",
        char_count=char_count,
        word_count=word_count,
        has_large_image=has_large_image,
        confidence=0.95,
        reason=f"Rich native text layer detected ({char_count} chars, {word_count} words)."
    )
