"""Abstract base class for OCR engines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class OCRWord:
    """Single recognized word with its bounding box and confidence score.

    Attributes:
        text: The recognized text string.
        left: Left edge of the bounding box in pixels.
        top: Top edge of the bounding box in pixels.
        right: Right edge of the bounding box in pixels.
        bottom: Bottom edge of the bounding box in pixels.
        confidence: Recognition confidence in [0.0, 1.0].
    """

    text: str
    left: int
    top: int
    right: int
    bottom: int
    confidence: float  # 0.0 – 1.0

    @property
    def x_center(self) -> float:
        """Horizontal center of the bounding box."""
        return (self.left + self.right) / 2

    @property
    def y_center(self) -> float:
        """Vertical center of the bounding box."""
        return (self.top + self.bottom) / 2

    @property
    def width(self) -> int:
        """Width of the bounding box in pixels."""
        return self.right - self.left

    @property
    def height(self) -> int:
        """Height of the bounding box in pixels."""
        return self.bottom - self.top


@dataclass
class OCRResult:
    """Full OCR result for a single page.

    Attributes:
        words: List of all recognized words with bounding boxes.
        full_text: Entire page text joined as a single string.
        page_confidence: Mean confidence across all recognized words.
        engine: Name of the OCR engine that produced this result.
        processing_time_ms: Wall-clock time for recognition in milliseconds.
        image_shape: Shape of the input image as (height, width).
    """

    words: list[OCRWord]
    full_text: str
    page_confidence: float
    engine: str
    processing_time_ms: int
    image_shape: tuple[int, int]  # (height, width)


class OCREngine(ABC):
    """Abstract interface that every OCR engine implementation must satisfy.

    Subclasses must implement :meth:`recognize`, :meth:`is_available`, and the
    :attr:`name` property.  All implementations are expected to accept a
    preprocessed, grayscale uint8 NumPy array and return a fully-populated
    :class:`OCRResult`.
    """

    @abstractmethod
    def recognize(self, image: np.ndarray) -> OCRResult:
        """Run OCR on a preprocessed grayscale image.

        Args:
            image: A uint8 NumPy array of shape (H, W) containing a
                   preprocessed grayscale page image.

        Returns:
            An :class:`OCRResult` populated with all recognized words,
            page-level text, and metadata.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether this engine and its runtime dependencies are present.

        Returns:
            ``True`` if the engine can be used, ``False`` otherwise.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine identifier used in logging and reporting."""
