"""Remote OCR engine — delegates recognition to an external microservice.

This engine is designed for deployment environments (e.g. Vercel edge
functions) where neither Tesseract nor ONNX runtimes are available.  It
encodes the image as base64 JSON, POSTs it to a configurable service URL, and
parses the JSON response back into a :class:`~engine_base.OCRResult`.

Expected REST API contract
--------------------------
**Request** ``POST {service_url}/ocr``::

    {
        "image_b64": "<base64-encoded PNG>",
        "dpi": 250
    }

**Response** (HTTP 200)::

    {
        "words": [
            {
                "text": "HDFC",
                "left": 120,
                "top": 45,
                "right": 180,
                "bottom": 65,
                "confidence": 0.98
            },
            ...
        ]
    }

**Health check** ``GET {service_url}/health`` → HTTP 200 means service is up.
"""
from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Optional

import cv2
import numpy as np

from .engine_base import OCREngine, OCRResult, OCRWord

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 30
_IMAGE_DPI = 250


class RemoteOCREngine(OCREngine):
    """OCR engine that calls an external HTTP microservice.

    The service URL can be provided at construction time or via the
    ``OCR_SERVICE_URL`` environment variable (the factory function
    :func:`core.ocr.get_ocr_engine` handles that).

    Attributes:
        service_url: Base URL of the OCR service (no trailing slash).
        timeout_s: HTTP request timeout in seconds (default 30).
    """

    def __init__(
        self,
        service_url: str,
        timeout_s: int = _DEFAULT_TIMEOUT_S,
    ) -> None:
        """Initialise the engine.

        Args:
            service_url: Base URL of the OCR REST service.  Must not have a
                         trailing slash (e.g. ``"https://ocr.example.com"``).
            timeout_s: Seconds before an HTTP request is considered timed out.
        """
        self.service_url = service_url.rstrip("/")
        self.timeout_s = timeout_s
        logger.info("RemoteOCREngine configured: service_url=%s", self.service_url)

    # ------------------------------------------------------------------
    # OCREngine interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Engine identifier."""
        return "remote"

    def is_available(self) -> bool:
        """Check whether the remote service is reachable and healthy.

        Performs a ``GET {service_url}/health`` request.  Returns ``True`` if
        the response status code is 200.

        Returns:
            ``True`` when the service is up; ``False`` on any error.
        """
        health_url = f"{self.service_url}/health"
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                available = resp.status == 200
                logger.debug(
                    "RemoteOCREngine health check %s → HTTP %d",
                    health_url, resp.status,
                )
                return available
        except Exception as exc:  # noqa: BLE001
            logger.debug("RemoteOCREngine health check failed: %s", exc)
            return False

    def recognize(self, image: np.ndarray) -> OCRResult:
        """POST a grayscale image to the remote OCR service and return results.

        The image is PNG-encoded and base64-encoded before transmission to keep
        the payload content-type simple (``application/json``).

        Args:
            image: Grayscale uint8 NumPy array (H, W).

        Returns:
            :class:`~engine_base.OCRResult` reconstructed from the service
            JSON response.

        Raises:
            RuntimeError: On HTTP errors, timeout, or invalid response JSON.
        """
        if image.ndim != 2:
            raise ValueError(
                f"RemoteOCREngine.recognize expects a 2-D grayscale array, "
                f"got shape {image.shape}"
            )

        h, w = image.shape
        logger.info(
            "RemoteOCREngine: sending %dx%d image to %s/ocr", w, h, self.service_url
        )

        # Encode image → PNG bytes → base64 string
        image_b64 = _encode_image_base64(image)

        payload = json.dumps({"image_b64": image_b64, "dpi": _IMAGE_DPI}).encode(
            "utf-8"
        )

        ocr_url = f"{self.service_url}/ocr"
        req = urllib.request.Request(
            ocr_url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw_body = resp.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Remote OCR service returned HTTP {exc.code}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Remote OCR service unreachable ({self.service_url}): {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Remote OCR service timed out after {self.timeout_s}s"
            ) from exc
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # Parse response
        try:
            response_data = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Remote OCR service returned invalid JSON: {exc}"
            ) from exc

        words = _parse_service_response(response_data)
        full_text = " ".join(word.text for word in words)
        page_confidence = (
            float(np.mean([word.confidence for word in words])) if words else 0.0
        )

        logger.info(
            "RemoteOCREngine: %d words, page_conf=%.3f, %d ms",
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


def _encode_image_base64(image: np.ndarray) -> str:
    """PNG-encode *image* and return a base64 string.

    Args:
        image: Grayscale or colour uint8 NumPy array.

    Returns:
        Base64-encoded PNG bytes as a plain ASCII string.

    Raises:
        RuntimeError: If OpenCV fails to encode the image.
    """
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to PNG-encode image for remote OCR service")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _parse_service_response(data: dict) -> list[OCRWord]:
    """Extract :class:`~engine_base.OCRWord` objects from the service JSON response.

    Expected structure::

        {"words": [{"text": ..., "left": ..., "top": ...,
                    "right": ..., "bottom": ..., "confidence": ...}, ...]}

    Args:
        data: Parsed JSON response dictionary.

    Returns:
        List of :class:`~engine_base.OCRWord` objects.  Malformed individual
        entries are skipped with a debug-level log.
    """
    words: list[OCRWord] = []
    raw_words = data.get("words", [])

    if not isinstance(raw_words, list):
        logger.warning("Remote OCR response 'words' field is not a list")
        return words

    for entry in raw_words:
        try:
            words.append(
                OCRWord(
                    text=str(entry["text"]).strip(),
                    left=int(entry["left"]),
                    top=int(entry["top"]),
                    right=int(entry["right"]),
                    bottom=int(entry["bottom"]),
                    confidence=float(entry["confidence"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Skipping malformed word entry from remote service: %s", exc)

    return words
