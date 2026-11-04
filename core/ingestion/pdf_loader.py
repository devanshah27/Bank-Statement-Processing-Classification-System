"""PDF loader module for Bank Statement Processing & Classification System.

Safely loads PDF documents using PyMuPDF (fitz) and pdfplumber, handling encryption,
passwords, and corrupt/truncated streams gracefully.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional, Any
import fitz  # PyMuPDF
import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class PDFLoadResult:
    """Result of loading a PDF file."""
    success: bool
    doc: Optional[fitz.Document] = None
    plumber_doc: Optional[pdfplumber.PDF] = None
    is_encrypted: bool = False
    needs_password: bool = False
    is_authenticated: bool = False
    page_count: int = 0
    file_size_bytes: int = 0
    error_message: Optional[str] = None


def load_pdf(file_bytes: bytes, password: Optional[str] = None) -> PDFLoadResult:
    """Load a PDF document from raw bytes.

    Attempts to open with PyMuPDF and pdfplumber. Handles password protection,
    invalid passwords, and corrupted streams without raising unhandled exceptions.

    Args:
        file_bytes: Raw bytes of the PDF file.
        password: Optional decryption password.

    Returns:
        PDFLoadResult with document handles or error information.
    """
    file_size = len(file_bytes)
    if not file_bytes:
        return PDFLoadResult(
            success=False,
            file_size_bytes=0,
            error_message="The provided PDF file is empty (0 bytes)."
        )

    # PyMuPDF attempt
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        logger.warning("Failed to open PDF stream with PyMuPDF: %s", e)
        return PDFLoadResult(
            success=False,
            file_size_bytes=file_size,
            error_message=f"Corrupted or invalid PDF structure: {str(e)}"
        )

    is_encrypted = bool(doc.is_encrypted)
    needs_password = False
    is_authenticated = True

    if is_encrypted:
        if password:
            auth_ok = doc.authenticate(password)
            if not auth_ok:
                doc.close()
                return PDFLoadResult(
                    success=False,
                    is_encrypted=True,
                    needs_password=True,
                    file_size_bytes=file_size,
                    error_message="Incorrect password provided for encrypted PDF."
                )
            is_authenticated = True
        else:
            # Check if it needs a password to read contents
            # PyMuPDF: doc.needs_pass is True if user password is required
            if doc.needs_pass:
                doc.close()
                return PDFLoadResult(
                    success=False,
                    is_encrypted=True,
                    needs_password=True,
                    file_size_bytes=file_size,
                    error_message="Password is required to decrypt this document."
                )

    page_count = len(doc)
    if page_count == 0:
        doc.close()
        return PDFLoadResult(
            success=False,
            file_size_bytes=file_size,
            error_message="PDF contains 0 pages."
        )

    # Open with pdfplumber for high-level table text extraction
    plumber_doc: Optional[pdfplumber.PDF] = None
    try:
        stream = io.BytesIO(file_bytes)
        plumber_doc = pdfplumber.open(stream, password=password)
    except Exception as e:
        err_str = str(e).lower()
        if "eof" in err_str or "damaged" in err_str or "corrupt" in err_str or "syntax" in err_str or getattr(doc, "is_repaired", False):
            doc.close()
            return PDFLoadResult(
                success=False,
                file_size_bytes=file_size,
                error_message=f"PDF file is truncated or corrupted: {e}",
            )
        logger.warning("pdfplumber failed to open stream (will fallback to fitz only): %s", e)
        plumber_doc = None

    return PDFLoadResult(
        success=True,
        doc=doc,
        plumber_doc=plumber_doc,
        is_encrypted=is_encrypted,
        needs_password=needs_password,
        is_authenticated=is_authenticated,
        page_count=page_count,
        file_size_bytes=file_size,
        error_message=None
    )
