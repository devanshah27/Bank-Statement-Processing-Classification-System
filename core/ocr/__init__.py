"""OCR Engine package and factory."""

from __future__ import annotations

import os
import logging
from typing import Optional
from .engine_base import OCREngine, OCRWord, OCRResult

logger = logging.getLogger(__name__)


_ENGINE_CACHE: dict[str, OCREngine] = {}


def get_ocr_engine(engine_name: Optional[str] = None) -> OCREngine:
    """Instantiate and return the configured OCR engine adapter (cached).

    Args:
        engine_name: "tesseract", "rapid", or "remote".
                     Defaults to OCR_ENGINE env var or "rapid".
    """
    raw_name = (engine_name or os.environ.get("OCR_ENGINE", "rapid")).lower().strip()
    if raw_name in ("tesseract", "tess"):
        canonical = "tesseract"
    elif raw_name == "remote":
        canonical = "remote"
    else:
        canonical = "rapid"

    if canonical not in _ENGINE_CACHE:
        if canonical == "tesseract":
            from .tesseract_engine import TesseractEngine
            _ENGINE_CACHE[canonical] = TesseractEngine()
        elif canonical == "remote":
            from .remote_engine import RemoteOCREngine
            service_url = os.environ.get("OCR_SERVICE_URL", "http://localhost:8080")
            _ENGINE_CACHE[canonical] = RemoteOCREngine(service_url)
        else:
            from .rapidocr_engine import RapidOCREngine
            _ENGINE_CACHE[canonical] = RapidOCREngine()

    return _ENGINE_CACHE[canonical]

