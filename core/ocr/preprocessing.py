"""OpenCV preprocessing pipeline for scanned bank statements.

The pipeline converts raw page images (photos, flatbed scans, fax output) into
clean, high-contrast, correctly oriented grayscale images suitable for OCR.

Typical call::

    result = preprocess_for_ocr(raw_image)
    ocr_text = engine.recognize(result.image)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .orientation import detect_orientation, rotate_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PreprocessResult:
    """Container for the preprocessed image and its provenance metadata.

    Attributes:
        image: Preprocessed grayscale uint8 NumPy array ready for OCR.
        original_shape: ``(height, width)`` of the input image before any
                        processing.
        skew_angle: Small-angle skew detected and corrected (degrees).
        rotation_applied: Coarse rotation applied to fix orientation
                          (0, 90, 180, or 270 degrees).
        was_deskewed: ``True`` if a skew correction was applied.
        was_shadow_removed: ``True`` if shadow / uneven illumination was removed.
        was_contrast_enhanced: ``True`` if CLAHE contrast enhancement ran.
        is_binary: ``True`` if the input was detected as a 1-bit fax-style scan.
    """

    image: np.ndarray
    original_shape: tuple[int, int]
    skew_angle: float
    rotation_applied: int
    was_deskewed: bool
    was_shadow_removed: bool
    was_contrast_enhanced: bool
    is_binary: bool


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def preprocess_for_ocr(image: np.ndarray, binarize: bool = False) -> PreprocessResult:
    """Run the optimized preprocessing pipeline on a raw page image.

    Steps (in order):
    1. Convert to grayscale.
    2. Detect and fix 90° orientation via projection-profile variance.
    3. Deskew small tilts (>= 0.5°).
    4. Remove shadow / uneven illumination (only when uneven illumination is detected).
    5. Denoise (noise-level adaptive).
    6. Contrast enhancement (CLAHE if low contrast).
    7. Adaptive thresholding (applied if binarize=True or near-binary low contrast).
    8. Detect and handle fax-style 1-bit low-resolution scans.

    Args:
        image: Raw page image as a NumPy array. May be colour (H, W, 3),
               colour with alpha (H, W, 4), or already grayscale (H, W).
               Expected dtype: uint8.
        binarize: Whether to force 1-bit adaptive threshold binarization.
                  Default is False (preserves grayscale anti-aliasing for DL OCR).

    Returns:
        A :class:`PreprocessResult` containing the cleaned image and flags
        describing which corrections were applied.
    """
    original_shape = (image.shape[0], image.shape[1])
    logger.info("Preprocessing image of shape %s", image.shape)

    # --- Step 1: Grayscale ---------------------------------------------------
    gray = _to_grayscale(image)

    # --- Step 2: Orientation -------------------------------------------------
    gray, rotation_applied = detect_and_fix_orientation(gray)
    logger.debug("Rotation applied: %d°", rotation_applied)

    # --- Step 3: Deskew ------------------------------------------------------
    gray, skew_angle = deskew(gray)
    was_deskewed = abs(skew_angle) >= 0.5
    logger.debug("Skew angle corrected: %.2f°", skew_angle)

    # --- Detect fax scan early so later steps can branch ---------------------
    is_binary = _is_binary_image(gray)

    # --- Step 4: Shadow removal (skip for 1-bit or uniformly illuminated scans) ---
    was_shadow_removed = False
    if not is_binary and not _is_shadow_free(gray):
        shadow_removed = remove_shadow_and_normalize(gray)
        # Only accept if it looks meaningfully different (avoids over-processing)
        if _rms_diff(gray, shadow_removed) > 2.0:
            gray = shadow_removed
            was_shadow_removed = True
            logger.debug("Shadow removal applied")

    # --- Step 5: Denoise -----------------------------------------------------
    gray = denoise(gray)

    # --- Step 6: Contrast enhancement (before thresholding if low contrast) --
    was_contrast_enhanced = False
    if not is_binary and _is_low_contrast(gray):
        gray = enhance_contrast(gray)
        was_contrast_enhanced = True
        logger.debug("CLAHE contrast enhancement applied")

    # --- Step 7: Adaptive threshold ------------------------------------------
    if binarize or (is_binary and _is_low_contrast(gray)):
        gray = adaptive_threshold(gray)

    # --- Step 8: Fax scan handling ------------------------------------------
    if is_binary:
        gray = handle_fax_scan(gray)
        logger.debug("Fax scan post-processing applied")

    logger.info("Preprocessing complete — binary=%s, shadow=%s, deskewed=%s",
                is_binary, was_shadow_removed, was_deskewed)

    return PreprocessResult(
        image=gray,
        original_shape=original_shape,
        skew_angle=skew_angle,
        rotation_applied=rotation_applied,
        was_deskewed=was_deskewed,
        was_shadow_removed=was_shadow_removed,
        was_contrast_enhanced=was_contrast_enhanced,
        is_binary=is_binary,
    )


# ---------------------------------------------------------------------------
# Individual processing steps
# ---------------------------------------------------------------------------


def detect_and_fix_orientation(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Detect the page orientation and rotate the image to upright.

    Bank statements are standard portrait documents (H >= W). If an input scan
    is already in portrait orientation, we preserve it to prevent destructive
    90° flips (fine line orientation is handled by the OCR angle classifier).
    If an input scan is landscape (W > H), it is rotated to portrait.

    Args:
        image: Grayscale uint8 image (H, W).

    Returns:
        Tuple of ``(corrected_image, degrees_rotated)`` where
        *degrees_rotated* ∈ {0, 90, 180, 270}.
    """
    h, w = image.shape[:2]
    if h >= w:
        return image, 0

    # For sideways scans (W > H), rotate to portrait (either 90° CW or 270° CW)
    rot90 = rotate_image(image, 90)
    rot270 = rotate_image(image, 270)
    inv90 = 255 - rot90
    inv270 = 255 - rot270
    diff90 = float(inv90[:int(0.3 * rot90.shape[0])].mean() - inv90[int(0.7 * rot90.shape[0]):].mean())
    diff270 = float(inv270[:int(0.3 * rot270.shape[0])].mean() - inv270[int(0.7 * rot270.shape[0]):].mean())

    if diff270 >= diff90:
        return rot270, 270
    return rot90, 90


def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Correct small rotational skew in a grayscale page image.

    Uses the Hough probabilistic line transform on a Canny edge image to find
    the dominant line angle, then corrects skew in the range [−15°, +15°].
    If no reliable angle is found (or if the image is very small), the original
    image is returned unchanged.

    Args:
        image: Grayscale uint8 image (H, W).

    Returns:
        Tuple of ``(deskewed_image, angle_in_degrees)`` where the angle is the
        skew that was corrected.  Positive = clockwise tilt.
    """
    h, w = image.shape[:2]
    if h < 50 or w < 50:
        logger.debug("Image too small for deskew (%dx%d), skipping", w, h)
        return image, 0.0

    # Edge detection on a down-scaled copy for speed
    scale = min(1.0, 1200.0 / max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(small.shape[1] * 0.3),
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        logger.debug("No Hough lines found — skew undetectable")
        return image, 0.0

    angles: list[float] = []
    for line in lines:
        coords = np.array(line).flatten()
        if len(coords) < 4:
            continue
        x1, y1, x2, y2 = coords[:4]
        if x2 == x1:
            continue
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        # Only consider near-horizontal lines
        if -15.0 < angle < 15.0:
            angles.append(angle)

    if not angles:
        logger.debug("No near-horizontal Hough lines — skew undetectable")
        return image, 0.0

    # Median angle is more robust to outliers than mean
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return image, 0.0

    logger.debug("Deskewing by %.2f°", median_angle)
    center = (w / 2, h / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(
        image,
        rot_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return deskewed, median_angle


def remove_shadow_and_normalize(image: np.ndarray) -> np.ndarray:
    """Remove shadow gradients and normalize illumination.

    Estimates the background using morphological dilation with a large kernel
    (which fills in dark text pixels), then divides the original image by this
    background estimate.  This corrects for phone-photo shadows and flatbed
    scanner illumination fall-off.

    Args:
        image: Grayscale uint8 image (H, W).

    Returns:
        Shadow-corrected and normalized grayscale uint8 image.
    """
    # Large dilation kernel blurs away text, leaving only background illumination
    kernel_size = max(15, min(image.shape[0], image.shape[1]) // 20 | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    background = cv2.dilate(image, kernel)
    # Smooth the background estimate
    background = cv2.GaussianBlur(background, (kernel_size | 1, kernel_size | 1), 0)

    # Subtract and normalize: result = (img / background) * 255
    img_f = image.astype(np.float32)
    bg_f = background.astype(np.float32)
    # Avoid division by zero
    bg_f = np.where(bg_f < 1.0, 1.0, bg_f)
    normalized = (img_f / bg_f) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return normalized


def denoise(image: np.ndarray) -> np.ndarray:
    """Apply noise reduction appropriate to the detected noise level.

    Noise level is estimated by computing the variance in a set of homogeneous
    (smooth) regions of the image.  Three denoising regimes are defined:

    * **Low noise** (σ < 5): image returned as-is.
    * **Moderate noise** (σ 5–15): bilateral filter that blurs noise while
      preserving sharp edges.
    * **High noise** (σ > 15): Gaussian blur followed by morphological closing
      to reconnect broken strokes.

    Args:
        image: Grayscale uint8 image (H, W).

    Returns:
        Denoised grayscale uint8 image.
    """
    noise_level = _estimate_noise(image)
    logger.debug("Estimated noise level (σ): %.2f", noise_level)

    if noise_level < 5.0:
        return image

    if noise_level < 15.0:
        # Bilateral filter — preserves edges (text strokes)
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # High noise: Gaussian + morphological closing
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Binarize the image using an illumination-adaptive threshold.

    If the image is already binary (or near-binary), only a light
    morphological clean-up is applied.  Otherwise, Gaussian adaptive
    thresholding is used (better than Otsu for images with local illumination
    variation), followed by morphological opening to remove speckle noise.

    Args:
        image: Grayscale uint8 image (H, W).

    Returns:
        Binary uint8 image with values in {0, 255}.
    """
    if _is_binary_image(image):
        # Light cleanup: remove isolated noise pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Block size must be odd and ≥ 3
    block_size = 31
    binary = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        11,
    )
    # Remove salt-and-pepper noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Suitable for low-contrast flatbed scans where text and background have
    similar grey levels.  CLAHE is applied tile-by-tile to avoid over-amplifying
    noise in already-bright regions.

    Args:
        image: Grayscale uint8 image (H, W).

    Returns:
        Contrast-enhanced grayscale uint8 image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def handle_fax_scan(image: np.ndarray) -> np.ndarray:
    """Post-process a 1-bit (fax-style) scan to improve OCR accuracy.

    Fax scans at 150 DPI often have horizontal streaks, broken characters, and
    insufficient resolution.  This function:

    1. Applies morphological closing to reconnect broken character strokes.
    2. Applies morphological opening to remove horizontal fax streaks.
    3. Up-scales 2× with bicubic interpolation if the image is small
       (heuristic: width < 1200 px ≈ 150 DPI letter).

    Args:
        image: Binary or near-binary grayscale uint8 image (H, W).

    Returns:
        Cleaned (and possibly up-scaled) grayscale uint8 image.
    """
    # Reconnect broken character strokes
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(image, cv2.MORPH_CLOSE, close_kernel)

    # Remove horizontal fax streaks (long, thin horizontal noise)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, h_kernel)

    # Up-scale if resolution looks very low (< 150 DPI letter = ~1240 px wide)
    if cleaned.shape[1] < 1200:
        logger.info(
            "Fax scan: up-scaling from %dx%d to %dx%d",
            cleaned.shape[1], cleaned.shape[0],
            cleaned.shape[1] * 2, cleaned.shape[0] * 2,
        )
        cleaned = cv2.resize(
            cleaned,
            (cleaned.shape[1] * 2, cleaned.shape[0] * 2),
            interpolation=cv2.INTER_CUBIC,
        )

    return cleaned


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert *image* to single-channel grayscale uint8."""
    if image.ndim == 2:
        return image.astype(np.uint8) if image.dtype != np.uint8 else image
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0]
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _is_shadow_free(image: np.ndarray) -> bool:
    """Fast check whether document illumination is uniform across the page.

    Samples 5 key patches (four corners and center). If high-percentile
    background intensities match closely (variation < 25), illumination is uniform.
    """
    h, w = image.shape[:2]
    if h < 100 or w < 100:
        return True
    ph = min(50, h // 4)
    pw = min(50, w // 4)
    patches = [
        image[10:10 + ph, 10:10 + pw],
        image[10:10 + ph, w - pw - 10:w - 10],
        image[h - ph - 10:h - 10, 10:10 + pw],
        image[h - ph - 10:h - 10, w - pw - 10:w - 10],
        image[h // 2 - ph // 2:h // 2 + ph // 2, w // 2 - pw // 2:w // 2 + pw // 2],
    ]
    bg_levels = [float(np.percentile(p, 90)) for p in patches if p.size > 0]
    if not bg_levels:
        return True
    return (max(bg_levels) - min(bg_levels)) < 25.0


def _is_binary_image(image: np.ndarray, threshold: float = 0.95) -> bool:
    """Return ``True`` if *image* is effectively binary (only 0s and 255s)."""
    sub = image[::4, ::4]
    if sub.size == 0:
        return False
    near_binary = np.count_nonzero((sub == 0) | (sub == 255))
    return float(near_binary) / sub.size >= threshold


def _is_low_contrast(image: np.ndarray) -> bool:
    """Return ``True`` if the image has low contrast (std < 40)."""
    sub = image[::4, ::4]
    return float(np.std(sub)) < 40.0


def _estimate_noise(image: np.ndarray) -> float:
    """Estimate noise level on a subsampled grid for fast evaluation.

    Uses Laplacian variance on a 4x subsampled view, executing in <2ms.
    """
    sub = image[::4, ::4]
    if sub.size < 100:
        return 0.0
    laplacian = cv2.Laplacian(sub, cv2.CV_32F)
    sigma = float(np.sqrt(np.abs(np.mean(laplacian ** 2))))
    return sigma / 6.0


def _rms_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square pixel difference between two same-shaped uint8 arrays."""
    return float(np.sqrt(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)))
