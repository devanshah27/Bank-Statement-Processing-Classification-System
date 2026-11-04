"""Stateless chunked processing state for serverless compatibility.

:class:`ChunkContext` carries all inter-chunk state as a signed, compressed,
base64-encoded token so that no server-side session storage is required.
This makes the processing pipeline compatible with stateless environments
(e.g. Vercel serverless functions).

Security
--------
Tokens are HMAC-SHA256 signed with ``CHUNK_SIGNING_KEY``. Tampered or
truncated tokens raise :class:`ValueError` on deserialisation.

Usage::

    ctx = ChunkContext(total_pages=10, bank_profile_name="hdfc")
    token = ctx.serialize()

    # -- Later, in another request --
    ctx2 = ChunkContext.deserialize(token)
    ctx2.last_page_processed = 1
    new_token = ctx2.serialize()
"""
from __future__ import annotations

import json
import hmac
import zlib
import base64
import hashlib
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIGNING_KEY: bytes = os.environ.get(
    "CHUNK_SIGNING_KEY", "dev-signing-key-change-in-prod"
).encode("utf-8")

_HMAC_ALGO = "sha256"
_TOKEN_SEPARATOR = "."


# ---------------------------------------------------------------------------
# ChunkContext
# ---------------------------------------------------------------------------


@dataclass
class ChunkContext:
    """Carries all inter-chunk processing state as a serialisable token.

    Attributes
    ----------
    last_page_processed:
        0-based index of the last successfully processed page.
    total_pages:
        Total number of pages in the document.
    bank_profile_name:
        Slug of the bank profile being used (e.g. ``"hdfc"``, ``"generic"``).
    column_anchors:
        Mapping of column names to their detected x-offsets or indices,
        preserved across chunks so column detection runs only once.
    running_balance:
        The balance value from the last row of the previous chunk, used to
        validate continuity in the next chunk.
    last_row_date:
        ISO date string (``YYYY-MM-DD``) of the last processed transaction,
        carried forward for year-inference on ambiguous date formats.
    account_details_extracted:
        ``True`` once account header details have been successfully parsed.
    partial_transactions:
        Serialised (dict) rows that span a chunk boundary and cannot be
        finalised until more pages are processed.
    statement_boundaries:
        Page numbers where a new statement begins (for multi-statement PDFs).
    is_complete:
        ``True`` when all pages have been processed.
    processing_time_ms:
        Cumulative wall-clock processing time in milliseconds.
    """

    last_page_processed: int = 0
    total_pages: int = 0
    bank_profile_name: str = "generic"
    column_anchors: dict[str, Any] = field(default_factory=dict)
    running_balance: Optional[float] = None
    last_row_date: Optional[str] = None
    account_details_extracted: bool = False
    partial_transactions: list[dict[str, Any]] = field(default_factory=list)
    statement_boundaries: list[int] = field(default_factory=list)
    is_complete: bool = False
    processing_time_ms: int = 0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def serialize(self) -> str:
        """Serialise this context to a signed, compressed token string.

        The token format is ``<hmac_hex>.<base64_zlib_json>``.

        Returns
        -------
        str
            Signed, compressed, base64-encoded token.

        Raises
        ------
        ValueError
            If the serialised payload exceeds an internal sanity limit
            (should not occur with normal data volumes).
        """
        payload_dict = asdict(self)
        json_bytes = json.dumps(payload_dict, separators=(",", ":")).encode("utf-8")
        compressed = zlib.compress(json_bytes, level=6)
        encoded = base64.urlsafe_b64encode(compressed).decode("ascii")
        signature = self._sign(encoded)
        token = f"{signature}{_TOKEN_SEPARATOR}{encoded}"
        logger.debug(
            "ChunkContext serialised: pages=%d/%d, token_len=%d",
            self.last_page_processed,
            self.total_pages,
            len(token),
        )
        return token

    @classmethod
    def deserialize(cls, token: str) -> "ChunkContext":
        """Deserialise and verify a token produced by :meth:`serialize`.

        Parameters
        ----------
        token:
            Token string in ``<hmac_hex>.<base64_payload>`` format.

        Returns
        -------
        ChunkContext

        Raises
        ------
        ValueError
            If the token is malformed, the HMAC signature is invalid, or
            the JSON payload cannot be parsed.
        """
        if not token or _TOKEN_SEPARATOR not in token:
            raise ValueError("ChunkContext token is malformed (missing separator)")

        sep_idx = token.index(_TOKEN_SEPARATOR)
        signature = token[:sep_idx]
        encoded = token[sep_idx + 1:]

        expected_sig = cls._sign(encoded)
        if not hmac.compare_digest(signature, expected_sig):
            raise ValueError("ChunkContext token HMAC verification failed — possible tampering")

        try:
            compressed = base64.urlsafe_b64decode(encoded.encode("ascii"))
            json_bytes = zlib.decompress(compressed)
            payload_dict = json.loads(json_bytes.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"ChunkContext token decompression/parse error: {exc}") from exc

        ctx = cls(
            last_page_processed=int(payload_dict.get("last_page_processed", 0)),
            total_pages=int(payload_dict.get("total_pages", 0)),
            bank_profile_name=str(payload_dict.get("bank_profile_name", "generic")),
            column_anchors=dict(payload_dict.get("column_anchors", {})),
            running_balance=payload_dict.get("running_balance"),
            last_row_date=payload_dict.get("last_row_date"),
            account_details_extracted=bool(payload_dict.get("account_details_extracted", False)),
            partial_transactions=list(payload_dict.get("partial_transactions", [])),
            statement_boundaries=list(payload_dict.get("statement_boundaries", [])),
            is_complete=bool(payload_dict.get("is_complete", False)),
            processing_time_ms=int(payload_dict.get("processing_time_ms", 0)),
        )
        logger.debug(
            "ChunkContext deserialised: pages=%d/%d, bank=%s",
            ctx.last_page_processed,
            ctx.total_pages,
            ctx.bank_profile_name,
        )
        return ctx

    @staticmethod
    def is_valid_token(token: str) -> bool:
        """Check whether a token has a valid HMAC signature without raising.

        Parameters
        ----------
        token:
            Token string to validate.

        Returns
        -------
        bool
            ``True`` if the signature is valid; ``False`` otherwise.
        """
        try:
            if not token or _TOKEN_SEPARATOR not in token:
                return False
            sep_idx = token.index(_TOKEN_SEPARATOR)
            signature = token[:sep_idx]
            encoded = token[sep_idx + 1:]
            expected_sig = ChunkContext._sign(encoded)
            return hmac.compare_digest(signature, expected_sig)
        except Exception:
            return False

    def progress_pct(self) -> float:
        """Return processing progress as a percentage (0.0–100.0).

        Returns
        -------
        float
            ``0.0`` if *total_pages* is zero or not yet set.
        """
        if not self.total_pages:
            return 0.0
        return min(100.0, (self.last_page_processed / self.total_pages) * 100.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sign(payload: str) -> str:
        """Compute HMAC-SHA256 hex digest of *payload* using the global key."""
        return hmac.new(
            SIGNING_KEY,
            payload.encode("ascii"),
            digestmod=hashlib.sha256,
        ).hexdigest()

    def __repr__(self) -> str:
        return (
            f"ChunkContext(pages={self.last_page_processed}/{self.total_pages}, "
            f"bank={self.bank_profile_name!r}, complete={self.is_complete}, "
            f"progress={self.progress_pct():.1f}%)"
        )
