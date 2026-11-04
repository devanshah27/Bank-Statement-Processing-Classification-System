"""Orientation detection without Tesseract OSD — Vercel-compatible.

Uses the horizontal projection profile variance method: for a correctly
oriented text image the row-sum projection exhibits high variance (alternating
bright inter-line gaps and dark text lines).  We score all four 90° rotations
and pick the one with the highest projection variance.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_orientation(image: np.ndarray) -> int:
    """Return the number of degrees to rotate *image* to reach correct orientation.

    The function evaluates all four 90-degree multiples (0, 90, 180, 270),
    computes the variance of the horizontal projection profile for each, and
    returns the rotation that maximises that variance — which corresponds to
    the orientation where text lines are horizontal and clearly separated.

    Args:
        image: Grayscale uint8 NumPy array of shape (H, W).  The image should
               already be binarized or at least contrast-enhanced so that text
               pixels are dark on a light background.

    Returns:
        One of ``{0, 90, 180, 270}`` — the degrees needed to bring the image
        to its natural (upright, left-to-right) orientation.
    """
    best_rotation: int = 0
    best_variance: float = -1.0

    candidate_rotations = [0, 90, 180, 270]

    for degrees in candidate_rotations:
        rotated = rotate_image(image, degrees)

        # Horizontal projection: sum of pixel values across each row.
        # For a grayscale image with dark text on white background, inverted so
        # text rows = high values → more variation between rows.
        inverted = 255 - rotated
        row_sums: np.ndarray = inverted.sum(axis=1).astype(np.float64)

        variance = float(np.var(row_sums))
        logger.debug(
            "Orientation candidate %d°: projection variance=%.2f", degrees, variance
        )

        if variance > best_variance:
            best_variance = variance
            best_rotation = degrees

    logger.info(
        "Detected orientation: rotate %d° (projection variance=%.2f)",
        best_rotation,
        best_variance,
    )
    return best_rotation


def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate *image* by an exact multiple of 90 degrees.

    Uses ``numpy.rot90`` for 90° multiples so no interpolation is applied and
    image quality is fully preserved.  Any other angle value raises a
    :class:`ValueError`.

    Args:
        image: Input NumPy array (any dtype, any number of channels).
        degrees: Rotation angle — must be one of ``{0, 90, 180, 270}``.

    Returns:
        The rotated image as a NumPy array.  The returned array may share
        memory with *image* when ``degrees == 0``.

    Raises:
        ValueError: If *degrees* is not a multiple of 90.
    """
    degrees = int(degrees) % 360

    if degrees not in (0, 90, 180, 270):
        raise ValueError(
            f"rotate_image only supports multiples of 90°, got {degrees}."
        )

    if degrees == 0:
        return image

    # numpy.rot90 rotates counter-clockwise; k=1 → 90° CCW.
    # We want clockwise rotation: 90° CW = 270° CCW (k=3), etc.
    k_map = {90: 3, 180: 2, 270: 1}
    k = k_map[degrees]
    return np.ascontiguousarray(np.rot90(image, k=k))
