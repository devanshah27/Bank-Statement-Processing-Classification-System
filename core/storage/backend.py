"""Storage backend interface and implementations.

Provides a pluggable storage abstraction:

- :class:`StorageBackend` — abstract base class defining the interface.
- :class:`StatelessBackend` — embeds data inside a signed, compressed token
  (zero server-side state; suitable for Vercel/edge deployments).
- :class:`InMemoryBackend` — simple dict-based store for local development.

Select a backend via :func:`get_storage_backend` or the ``DEPLOY_TARGET``
environment variable.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import zlib
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SIGNING_KEY: bytes = os.environ.get(
    "STORAGE_SIGNING_KEY", "dev-storage-key-change-in-prod"
).encode("utf-8")

_HMAC_ALGO = "sha256"
_TOKEN_SEP = "."
_MAX_DATA_BYTES = 1 * 1024 * 1024  # 1 MB ceiling for StatelessBackend


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class StorageBackend(ABC):
    """Abstract interface for statement processing state storage.

    All implementations must be safe for concurrent use (stateless or
    appropriately locked).
    """

    @abstractmethod
    def save(self, key: str, data: dict) -> str:
        """Persist *data* under *key* and return a retrieval token.

        Parameters
        ----------
        key:
            Logical identifier for the stored object (e.g. a session ID).
        data:
            JSON-serialisable dict to store.

        Returns
        -------
        str
            An opaque token that can be passed to :meth:`load` to retrieve
            the data, possibly in a different process or request.
        """

    @abstractmethod
    def load(self, token: str) -> Optional[dict]:
        """Load and return data for the given *token*.

        Parameters
        ----------
        token:
            Token returned by a previous :meth:`save` call.

        Returns
        -------
        dict or ``None``
            The stored data, or ``None`` if the token is unknown.

        Raises
        ------
        ValueError
            If the token is structurally invalid or its integrity check fails.
        """

    @abstractmethod
    def delete(self, token: str) -> None:
        """Remove data associated with *token* (no-op if not found).

        Parameters
        ----------
        token:
            Token returned by a previous :meth:`save` call.
        """


# ---------------------------------------------------------------------------
# Stateless backend (token-embedded data)
# ---------------------------------------------------------------------------


class StatelessBackend(StorageBackend):
    """Stores data entirely within a signed, compressed token.

    No server-side session storage is required. The token format is::

        <hmac_hex>.<base64url(zlib(json(data)))>

    Data is verified on load with HMAC-SHA256. Tampered tokens raise
    :class:`ValueError`.

    Limitations
    -----------
    - Tokens grow with data size; keep data small.
    - Delete is a no-op (no server-side state to remove).
    - ``key`` is embedded in the token payload for round-trip verification
      but is not used for routing.

    Parameters
    ----------
    signing_key:
        Raw bytes used for HMAC signing. Defaults to the ``STORAGE_SIGNING_KEY``
        environment variable (falls back to a development placeholder).
    max_data_bytes:
        Maximum uncompressed JSON size in bytes. Defaults to 1 MB.
    """

    def __init__(
        self,
        signing_key: Optional[bytes] = None,
        max_data_bytes: int = _MAX_DATA_BYTES,
    ) -> None:
        self._key: bytes = signing_key or _SIGNING_KEY
        self._max_bytes = max_data_bytes

    # ------------------------------------------------------------------

    def save(self, key: str, data: dict) -> str:
        """Serialise, compress, sign, and return *data* as a self-contained token.

        Parameters
        ----------
        key:
            Logical key; embedded in the payload for verification on load.
        data:
            JSON-serialisable dict.

        Returns
        -------
        str
            Opaque signed token.

        Raises
        ------
        ValueError
            If the serialised data exceeds :attr:`_max_bytes`.
        TypeError
            If *data* is not JSON-serialisable.
        """
        envelope = {"_key": key, "data": data}
        json_bytes = json.dumps(envelope, separators=(",", ":")).encode("utf-8")

        if len(json_bytes) > self._max_bytes:
            raise ValueError(
                f"StatelessBackend: data for key '{key}' is {len(json_bytes):,} bytes, "
                f"which exceeds the {self._max_bytes:,}-byte limit. "
                "Consider using InMemoryBackend or an external store for large datasets."
            )

        compressed = zlib.compress(json_bytes, level=6)
        encoded = base64.urlsafe_b64encode(compressed).decode("ascii")
        signature = self._sign(encoded)
        token = f"{signature}{_TOKEN_SEP}{encoded}"

        logger.debug(
            "StatelessBackend.save: key='%s', raw=%d B, compressed=%d B, token_len=%d",
            key, len(json_bytes), len(compressed), len(token),
        )
        return token

    def load(self, token: str) -> Optional[dict]:
        """Verify and deserialise a token returned by :meth:`save`.

        Parameters
        ----------
        token:
            Token string in ``<hmac>.<payload>`` format.

        Returns
        -------
        dict or ``None``
            The original *data* dict, or ``None`` if *token* is empty.

        Raises
        ------
        ValueError
            If the token is malformed, the HMAC is invalid, or JSON parsing fails.
        """
        if not token:
            return None

        if _TOKEN_SEP not in token:
            raise ValueError("StatelessBackend: token is malformed (missing separator)")

        sep_idx = token.index(_TOKEN_SEP)
        signature = token[:sep_idx]
        encoded = token[sep_idx + 1:]

        expected = self._sign(encoded)
        if not hmac.compare_digest(signature, expected):
            raise ValueError(
                "StatelessBackend: HMAC verification failed — token may be tampered or corrupted"
            )

        try:
            compressed = base64.urlsafe_b64decode(encoded.encode("ascii"))
            json_bytes = zlib.decompress(compressed)
            envelope = json.loads(json_bytes.decode("utf-8"))
        except Exception as exc:
            raise ValueError(
                f"StatelessBackend: failed to decompress/parse token: {exc}"
            ) from exc

        logger.debug("StatelessBackend.load: key='%s'", envelope.get("_key", "<unknown>"))
        return envelope.get("data")

    def delete(self, token: str) -> None:
        """No-op — stateless tokens cannot be invalidated server-side.

        Parameters
        ----------
        token:
            Ignored.
        """
        logger.debug("StatelessBackend.delete: no-op for stateless token (len=%d)", len(token))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sign(self, payload: str) -> str:
        """Compute HMAC-SHA256 hex digest of *payload*."""
        return hmac.new(
            self._key,
            payload.encode("ascii"),
            digestmod=hashlib.sha256,
        ).hexdigest()


# ---------------------------------------------------------------------------
# In-memory backend (local development)
# ---------------------------------------------------------------------------


class InMemoryBackend(StorageBackend):
    """Simple dict-based storage for local development and testing.

    Data is kept in process memory only; it does not survive restarts.
    Not suitable for multi-process or distributed deployments.
    """

    _shared_store: dict[str, dict] = {}

    def __init__(self) -> None:
        self._store = self._shared_store

    def save(self, key: str, data: dict) -> str:
        """Store *data* under *key* and return *key* as the retrieval token.

        Parameters
        ----------
        key:
            Logical key / session identifier.
        data:
            Any dict (need not be JSON-serialisable for in-memory use).

        Returns
        -------
        str
            The *key* itself, used as the retrieval token.
        """
        self._store[key] = data
        logger.debug("InMemoryBackend.save: key='%s', entries=%d", key, len(self._store))
        return key

    def load(self, token: str) -> Optional[dict]:
        """Return the data stored under *token* (== the original key).

        Parameters
        ----------
        token:
            Key returned by :meth:`save`.

        Returns
        -------
        dict or ``None``
            Stored data, or ``None`` if the key is unknown.
        """
        result = self._store.get(token)
        if result is None:
            logger.debug("InMemoryBackend.load: key='%s' not found", token)
        return result

    def delete(self, token: str) -> None:
        """Remove data stored under *token*.

        Parameters
        ----------
        token:
            Key returned by :meth:`save`.
        """
        removed = self._store.pop(token, None)
        if removed is not None:
            logger.debug("InMemoryBackend.delete: key='%s' removed", token)
        else:
            logger.debug("InMemoryBackend.delete: key='%s' not found (no-op)", token)

    def __len__(self) -> int:
        """Return the number of items currently in the store."""
        return len(self._store)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SHARED_IN_MEMORY_BACKEND: Optional[InMemoryBackend] = None

def get_storage_backend(deploy_target: Optional[str] = None) -> StorageBackend:
    """Return the appropriate :class:`StorageBackend` for the deployment target.

    Resolution order:

    1. The *deploy_target* parameter (if provided).
    2. The ``DEPLOY_TARGET`` environment variable.
    3. Falls back to :class:`InMemoryBackend` (singleton).

    Parameters
    ----------
    deploy_target:
        One of ``"vercel"``, ``"local"``, or ``None``.

    Returns
    -------
    :class:`StorageBackend`
        Configured backend instance.
    """
    global _SHARED_IN_MEMORY_BACKEND
    target = (deploy_target or os.environ.get("DEPLOY_TARGET", "local")).lower().strip()

    if target == "vercel":
        logger.info("Storage backend: StatelessBackend (Vercel/serverless mode)")
        return StatelessBackend()

    if target == "local":
        if _SHARED_IN_MEMORY_BACKEND is None:
            _SHARED_IN_MEMORY_BACKEND = InMemoryBackend()
        return _SHARED_IN_MEMORY_BACKEND

    # Unknown target — default to stateless (safe for any environment)
    logger.warning(
        "Unknown DEPLOY_TARGET '%s'; defaulting to StatelessBackend", target
    )
    return StatelessBackend()
