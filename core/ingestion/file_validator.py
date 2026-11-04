"""File validation module for the Bank Statement Processing & Classification System.

Validates uploaded files by checking magic bytes (file signatures), MIME type,
and enforcing size limits appropriate to the deployment target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Magic byte signatures
# ---------------------------------------------------------------------------
PDF_MAGIC: bytes = b"%PDF-"
PNG_MAGIC: bytes = b"\x89PNG"
JPEG_MAGIC: bytes = b"\xff\xd8\xff"
ZIP_MAGIC: bytes = b"PK\x03\x04"      # ZIP / XLSX / DOCX etc.
TIFF_MAGIC_LE: bytes = b"II\x2a\x00"  # TIFF little-endian
TIFF_MAGIC_BE: bytes = b"MM\x00\x2a"  # TIFF big-endian
GIF_MAGIC: bytes = b"GIF8"
WEBP_MAGIC: bytes = b"RIFF"           # WEBP starts with RIFF....WEBP

# ---------------------------------------------------------------------------
# Size limits (bytes)
# ---------------------------------------------------------------------------
_SIZE_LIMITS: dict[str, int] = {
    "vercel": 4 * 1024 * 1024,   # 4 MB on Vercel
    "local": 50 * 1024 * 1024,   # 50 MB locally
}
_DEFAULT_SIZE_LIMIT: int = 50 * 1024 * 1024

# ---------------------------------------------------------------------------
# Allowed file extensions → canonical MIME-ish type
# ---------------------------------------------------------------------------
_ALLOWED_EXTENSIONS: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


class ErrorCode(str, Enum):
    """Enumeration of validation error codes."""

    NONE = "NONE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_EXTENSION = "UNSUPPORTED_EXTENSION"
    INVALID_MAGIC_BYTES = "INVALID_MAGIC_BYTES"
    EXTENSION_MIME_MISMATCH = "EXTENSION_MIME_MISMATCH"
    EMPTY_FILE = "EMPTY_FILE"
    ENCRYPTED_ZIP = "ENCRYPTED_ZIP"


@dataclass
class ValidationResult:
    """Result returned by :func:`validate_file`.

    Attributes:
        is_valid:       ``True`` when the file passes all checks.
        error_code:     Machine-readable error identifier (``NONE`` on success).
        error_message:  Human-readable explanation of the validation failure.
        detected_type:  MIME-like type inferred from magic bytes
                        (e.g. ``"application/pdf"``, ``"image/png"``).
        file_size:      Size of the file in bytes.
    """

    is_valid: bool
    error_code: ErrorCode
    error_message: str
    detected_type: Optional[str]
    file_size: int = field(default=0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_type_from_magic(header: bytes) -> Optional[str]:
    """Return a MIME-like type string by inspecting the first few bytes.

    Args:
        header: At least the first 12 bytes of the file.

    Returns:
        A MIME-like string such as ``"application/pdf"`` or ``None`` if
        the signature is not recognised.
    """
    if header[:5] == PDF_MAGIC:
        return "application/pdf"
    if header[:4] == PNG_MAGIC:
        return "image/png"
    if header[:3] == JPEG_MAGIC:
        return "image/jpeg"
    if header[:4] in (TIFF_MAGIC_LE, TIFF_MAGIC_BE):
        return "image/tiff"
    if header[:4] == GIF_MAGIC:
        return "image/gif"
    if header[:4] == ZIP_MAGIC:
        # Distinguish plain ZIP from encrypted ZIP (flag bit 0 set in local file header)
        return "application/zip"
    if header[:4] == WEBP_MAGIC and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def _extension_from_filename(filename: str) -> str:
    """Extract the lower-cased file extension including the leading dot.

    Args:
        filename: Original filename string (may include path components).

    Returns:
        Extension such as ``".pdf"`` or ``""`` if none found.
    """
    dot_idx = filename.rfind(".")
    if dot_idx == -1:
        return ""
    return filename[dot_idx:].lower()


def _size_limit_for_target(deploy_target: str) -> int:
    """Return the byte-level file size limit for *deploy_target*.

    Args:
        deploy_target: Deployment environment string (``"vercel"`` or ``"local"``).

    Returns:
        Maximum allowed file size in bytes.
    """
    return _SIZE_LIMITS.get(deploy_target.lower(), _DEFAULT_SIZE_LIMIT)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_file(
    file_bytes: bytes,
    filename: str,
    deploy_target: str = "local",
) -> ValidationResult:
    """Validate an uploaded file for processing.

    Checks performed (in order):
    1. File is not empty.
    2. File size is within the allowed limit for the deployment target.
    3. File extension is in the allow-list.
    4. Magic bytes match a supported file type.
    5. Magic-byte-detected type is consistent with the file extension.
    6. ZIP files are not password-protected (a zip with encryption flag set
       cannot be extracted without a password and is rejected).

    Args:
        file_bytes:    Raw bytes of the uploaded file.
        filename:      Original filename provided by the client (used for
                       extension detection and log context only — do **not**
                       trust it for security decisions; magic bytes take
                       precedence).
        deploy_target: Deployment environment (``"local"`` or ``"vercel"``).
                       Controls the size limit applied.

    Returns:
        A :class:`ValidationResult` describing the outcome.

    Examples:
        >>> with open("statement.pdf", "rb") as fh:
        ...     data = fh.read()
        >>> result = validate_file(data, "statement.pdf", "local")
        >>> result.is_valid
        True
    """
    file_size = len(file_bytes)

    logger.debug(
        "Validating file name=%r size=%d bytes deploy_target=%r",
        filename,
        file_size,
        deploy_target,
    )

    # ── 1. Empty file check ─────────────────────────────────────────────────
    if file_size == 0:
        logger.warning("Rejected empty file: name=%r", filename)
        return ValidationResult(
            is_valid=False,
            error_code=ErrorCode.EMPTY_FILE,
            error_message="The uploaded file is empty.",
            detected_type=None,
            file_size=file_size,
        )

    # ── 2. Size limit check ─────────────────────────────────────────────────
    limit = _size_limit_for_target(deploy_target)
    if file_size > limit:
        limit_mb = limit / (1024 * 1024)
        logger.warning(
            "Rejected oversized file: name=%r size=%d limit=%d bytes",
            filename,
            file_size,
            limit,
        )
        return ValidationResult(
            is_valid=False,
            error_code=ErrorCode.FILE_TOO_LARGE,
            error_message=(
                f"File size {file_size / (1024 * 1024):.2f} MB exceeds the "
                f"{limit_mb:.0f} MB limit for the '{deploy_target}' deployment."
            ),
            detected_type=None,
            file_size=file_size,
        )

    # ── 3. Extension allow-list ─────────────────────────────────────────────
    extension = _extension_from_filename(filename)
    if extension not in _ALLOWED_EXTENSIONS:
        logger.warning(
            "Rejected unsupported extension: name=%r ext=%r", filename, extension
        )
        allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
        return ValidationResult(
            is_valid=False,
            error_code=ErrorCode.UNSUPPORTED_EXTENSION,
            error_message=(
                f"File extension '{extension or '(none)'}' is not supported. "
                f"Allowed extensions: {allowed}."
            ),
            detected_type=None,
            file_size=file_size,
        )

    # ── 4. Magic byte detection ─────────────────────────────────────────────
    header = file_bytes[:12]
    detected_type = _detect_type_from_magic(header)

    if detected_type is None:
        logger.warning(
            "Rejected file with unrecognised magic bytes: name=%r header=%r",
            filename,
            header,
        )
        return ValidationResult(
            is_valid=False,
            error_code=ErrorCode.INVALID_MAGIC_BYTES,
            error_message=(
                "The file's internal signature does not match any supported format "
                "(PDF, PNG, JPEG, TIFF). The file may be corrupted or misnamed."
            ),
            detected_type=None,
            file_size=file_size,
        )

    # ── 5. Extension ↔ detected type consistency ────────────────────────────
    expected_mime = _ALLOWED_EXTENSIONS[extension]
    if detected_type != expected_mime:
        # Special case: ZIP magic may be an XLSX masquerading as something else
        logger.warning(
            "Extension/magic mismatch: name=%r ext=%r expected=%r detected=%r",
            filename,
            extension,
            expected_mime,
            detected_type,
        )
        return ValidationResult(
            is_valid=False,
            error_code=ErrorCode.EXTENSION_MIME_MISMATCH,
            error_message=(
                f"File extension '{extension}' suggests {expected_mime}, but the "
                f"file content was detected as {detected_type}. "
                "Please upload the correct file."
            ),
            detected_type=detected_type,
            file_size=file_size,
        )

    # ── 6. ZIP encryption check (future-proofing) ───────────────────────────
    if detected_type == "application/zip" and len(file_bytes) >= 30:
        # Local file header general purpose bit flag is at bytes 6-7
        gp_flag = int.from_bytes(file_bytes[6:8], byteorder="little")
        if gp_flag & 0x1:  # bit 0 set → encrypted
            logger.warning("Rejected encrypted ZIP: name=%r", filename)
            return ValidationResult(
                is_valid=False,
                error_code=ErrorCode.ENCRYPTED_ZIP,
                error_message="Encrypted (password-protected) ZIP files are not supported.",
                detected_type=detected_type,
                file_size=file_size,
            )

    logger.info(
        "File validated successfully: name=%r type=%r size=%d bytes",
        filename,
        detected_type,
        file_size,
    )
    return ValidationResult(
        is_valid=True,
        error_code=ErrorCode.NONE,
        error_message="",
        detected_type=detected_type,
        file_size=file_size,
    )
