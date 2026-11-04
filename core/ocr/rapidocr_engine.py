"""RapidOCR (ONNX-based) engine — Vercel-compatible, no Tesseract binary needed.

RapidOCR is a lightweight OCR library based on PaddleOCR models exported to
ONNX.  It runs entirely in-process via ``onnxruntime``, making it suitable for
serverless environments.

The import of ``rapidocr_onnxruntime`` is lazy — importing this module is safe
even when the package is not installed.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from .engine_base import OCREngine, OCRResult, OCRWord

logger = logging.getLogger(__name__)

_SHARED_RAPID_OCR: Optional[Any] = None


class RapidOCREngine(OCREngine):
    """OCR engine backed by ``rapidocr_onnxruntime``.

    RapidOCR operates at the *line* level: each detected text region is a
    quadrilateral bounding box covering a full line of text.  This engine
    normalises those line-level boxes into the word-level
    :class:`~engine_base.OCRWord` format by treating each line as a single
    "word" (a simplification that is acceptable for downstream table recovery
    which re-groups by y-position anyway).

    If the recognised text contains spaces, the line is split into individual
    tokens and each token is assigned an equal share of the line's bounding
    box width.

    Attributes:
        _ocr: Cached ``RapidOCR`` instance (reused across calls).
    """

    def __init__(self) -> None:
        """Initialise the engine without importing ``rapidocr_onnxruntime``."""
        global _SHARED_RAPID_OCR
        self._ocr = _SHARED_RAPID_OCR

    # ------------------------------------------------------------------
    # OCREngine interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Engine identifier."""
        return "rapidocr"

    def is_available(self) -> bool:
        """Return ``True`` if ``rapidocr_onnxruntime`` can be imported.

        Returns:
            ``True`` when the package (and its ONNX model files) are present.
        """
        try:
            import rapidocr_onnxruntime  # noqa: F401, PLC0415

            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("RapidOCR not available: %s", exc)
            return False

    def recognize(self, image: np.ndarray) -> OCRResult:
        """Run RapidOCR on a preprocessed grayscale image.

        RapidOCR expects an RGB or grayscale uint8 NumPy array.  Grayscale
        images are converted to RGB internally before inference.

        Args:
            image: Grayscale uint8 NumPy array (H, W).

        Returns:
            :class:`~engine_base.OCRResult` with word (or line) level data.

        Raises:
            RuntimeError: If ``rapidocr_onnxruntime`` is not installed or
                          inference fails.
        """
        try:
            from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "rapidocr_onnxruntime is not installed. "
                "Run: pip install rapidocr-onnxruntime"
            ) from exc

        if image.ndim != 2:
            raise ValueError(
                f"RapidOCREngine.recognize expects a 2-D grayscale array, "
                f"got shape {image.shape}"
            )

        h, w = image.shape
        logger.info("RapidOCREngine: recognizing %dx%d image", w, h)

        # Build or reuse the shared RapidOCR instance lazily
        global _SHARED_RAPID_OCR
        if self._ocr is None:
            if _SHARED_RAPID_OCR is None:
                logger.debug("Initializing RapidOCR with angle classification enabled...")
                _SHARED_RAPID_OCR = RapidOCR(use_angle_cls=True)
                # Pre-warm with a tiny blank array so first request doesn't pay warmup cost
                try:
                    _SHARED_RAPID_OCR(np.full((64, 64), 255, dtype=np.uint8))
                except Exception:
                    pass
            self._ocr = _SHARED_RAPID_OCR

        t0 = time.perf_counter()
        try:
            # Pass 2D grayscale directly — RapidOCR converts via fast C++ OpenCV
            result, _ = self._ocr(image)
        except Exception as exc:
            raise RuntimeError(f"RapidOCR inference failed: {exc}") from exc
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        words = _parse_rapidocr_result(result or [])
        full_text = " ".join(word.text for word in words)

        page_confidence = (
            float(np.mean([w.confidence for w in words])) if words else 0.0
        )

        logger.info(
            "RapidOCREngine: %d words/lines, page_conf=%.3f, %d ms",
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


def _parse_rapidocr_result(result: list[Any]) -> list[OCRWord]:
    """Convert RapidOCR output into a list of :class:`~engine_base.OCRWord`.

    RapidOCR returns a list of triples: ``(box, text, confidence)`` where
    *box* is a list of 4 ``[x, y]`` corner points (top-left, top-right,
    bottom-right, bottom-left).

    If the recognised text contains multiple tokens (spaces), each token is
    split and assigned an equal horizontal slice of the bounding box so that
    table recovery can correctly assign tokens to columns.

    Args:
        result: Raw output list from ``RapidOCR.__call__``.  May be ``None``
                or empty.

    Returns:
        Flat list of :class:`~engine_base.OCRWord` objects.
    """
    words: list[OCRWord] = []

    for entry in result:
        if entry is None or len(entry) < 3:
            continue

        box, text, confidence = entry[0], str(entry[1]).strip(), float(entry[2])

        if not text:
            continue

        # Derive axis-aligned bounding box from the quadrilateral corners
        try:
            xs = [float(pt[0]) for pt in box]
            ys = [float(pt[1]) for pt in box]
        except (TypeError, IndexError, ValueError) as exc:
            logger.debug("Skipping RapidOCR entry with invalid box: %s", exc)
            continue

        line_left = int(min(xs))
        line_top = int(min(ys))
        line_right = int(max(xs))
        line_bottom = int(max(ys))
        line_width = max(line_right - line_left, 1)

        tokens = text.split()
        if len(tokens) <= 1:
            # Single token — use full bounding box as-is
            words.append(
                OCRWord(
                    text=text,
                    left=line_left,
                    top=line_top,
                    right=line_right,
                    bottom=line_bottom,
                    confidence=confidence,
                )
            )
        else:
            # Multiple tokens — split the bbox horizontally
            token_width = line_width / len(tokens)
            for i, token in enumerate(tokens):
                token_left = line_left + int(i * token_width)
                token_right = line_left + int((i + 1) * token_width)
                words.append(
                    OCRWord(
                        text=token,
                        left=token_left,
                        top=line_top,
                        right=token_right,
                        bottom=line_bottom,
                        confidence=confidence,
                    )
                )

    return words
