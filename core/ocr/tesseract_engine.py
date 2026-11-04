"""Tesseract OCR engine implementation.

Wraps ``pytesseract`` (which in turn shells out to the Tesseract binary) and
converts its output into the unified :class:`~engine_base.OCRResult` format.

The import of ``pytesseract`` is lazy — it only happens when :meth:`recognize`
or :meth:`is_available` is called, so importing this module does not raise an
error when Tesseract is not installed.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from .engine_base import OCREngine, OCRResult, OCRWord

if TYPE_CHECKING:
    import pytesseract  # noqa: F401 — type-checking only

logger = logging.getLogger(__name__)

# Tesseract page segmentation mode 6: assume a single uniform block of text.
# PSM 3 (auto) can be used for mixed layouts; 6 is faster for statements.
_DEFAULT_PSM = 6
_TESSERACT_CONFIG = f"--oem 3 --psm {_DEFAULT_PSM}"


class TesseractEngine(OCREngine):
    """OCR engine backed by Tesseract via the ``pytesseract`` wrapper.

    Lazy-imports ``pytesseract`` on first use.  Recognition is performed by
    calling ``pytesseract.image_to_data`` with ``output_type=DataFrame``, then
    filtering out entries with ``conf < 0`` (Tesseract uses -1 for non-word
    rows such as block/paragraph/line entries) and words whose text is blank.

    Attributes:
        config: Tesseract CLI configuration string passed on every call.
        lang: Tesseract language code (default ``"eng"``).
    """

    _cached_available: Optional[bool] = None

    def __init__(
        self,
        config: str = _TESSERACT_CONFIG,
        lang: str = "eng",
    ) -> None:
        """Initialise the engine.

        Args:
            config: Tesseract CLI flags (e.g. ``"--oem 3 --psm 6"``).
            lang: Tesseract language code or ``+``-delimited combo.
        """
        self.config = config
        self.lang = lang
        self._pytesseract: Optional[object] = None  # cached import

    # ------------------------------------------------------------------
    # OCREngine interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Engine identifier."""
        return "tesseract"

    def is_available(self) -> bool:
        """Return ``True`` if the ``pytesseract`` package and Tesseract binary exist.

        Caches the result so external CLI process is called at most once.
        """
        if TesseractEngine._cached_available is not None:
            return TesseractEngine._cached_available
        try:
            import pytesseract  # noqa: PLC0415

            pytesseract.get_tesseract_version()
            TesseractEngine._cached_available = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tesseract not available: %s", exc)
            TesseractEngine._cached_available = False
        return TesseractEngine._cached_available

    def recognize(self, image: np.ndarray) -> OCRResult:
        """Run Tesseract on a preprocessed grayscale image.

        Uses ``image_to_data`` (TSV output) to retrieve word-level bounding
        boxes and confidence scores, with a 30s timeout guard.
        """
        try:
            import pytesseract  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "pytesseract is not installed. Run: pip install pytesseract"
            ) from exc

        if image.ndim != 2:
            raise ValueError(
                f"TesseractEngine.recognize expects a 2-D grayscale array, "
                f"got shape {image.shape}"
            )

        h, w = image.shape
        logger.info("TesseractEngine: recognizing %dx%d image", w, h)

        t0 = time.perf_counter()
        try:
            df = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=self.config,
                output_type=pytesseract.Output.DATAFRAME,
                timeout=30,
            )
        except Exception as exc:
            raise RuntimeError(f"Tesseract recognition failed: {exc}") from exc
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        words = _parse_tesseract_dataframe(df)
        full_text = " ".join(w.text for w in words)

        if words:
            confidences = [w.confidence for w in words]
            page_confidence = float(np.mean(confidences))
        else:
            page_confidence = 0.0

        logger.info(
            "TesseractEngine: %d words, page_conf=%.3f, %d ms",
            len(words), page_confidence, elapsed_ms,
        )

        return OCRResult(
            words=words,
            full_text=full_text,
            page_confidence=page_confidence,
            engine=self.name,
            processing_time_ms=elapsed_ms,
            image_shape=(h, w),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_tesseract_dataframe(df: "object") -> list[OCRWord]:  # type: ignore[type-arg]
    """Convert a pytesseract TSV DataFrame into a list of :class:`~engine_base.OCRWord`.

    Filters out:
    * Rows with ``conf == -1`` (block / paragraph / line level entries).
    * Rows where ``text`` is NaN or whitespace-only.

    Tesseract confidence values are integers in [0, 100]; they are normalised
    to [0.0, 1.0] here.

    Args:
        df: pandas DataFrame returned by ``pytesseract.image_to_data``.

    Returns:
        List of :class:`~engine_base.OCRWord` with one entry per word token.
    """
    # Import pandas lazily — it's a pytesseract transitive dependency
    import pandas as pd  # noqa: PLC0415

    words: list[OCRWord] = []

    # Tesseract uses -1 as sentinel for non-word rows
    word_rows = df[df["conf"] > 0].copy()

    # Drop rows where text is NaN or blank
    word_rows = word_rows.dropna(subset=["text"])
    word_rows = word_rows[word_rows["text"].str.strip() != ""]

    for _, row in word_rows.iterrows():
        try:
            left = int(row["left"])
            top = int(row["top"])
            width = int(row["width"])
            height = int(row["height"])
            confidence = float(row["conf"]) / 100.0
            text = str(row["text"]).strip()

            words.append(
                OCRWord(
                    text=text,
                    left=left,
                    top=top,
                    right=left + width,
                    bottom=top + height,
                    confidence=confidence,
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Skipping malformed Tesseract row: %s", exc)

    return words
