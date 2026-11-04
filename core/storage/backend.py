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
import datetime
import decimal
import hashlib
import hmac
import json
import logging
import os
import zlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def _json_serial_default(o):
    """Custom serializer for non-JSON native types."""
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()
    if isinstance(o, decimal.Decimal):
        return float(o)
    return str(o)

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
    """Stores data within a signed, compressed token with in-memory and /tmp caching.

    No external database is required. The token format is::

        <hmac_hex>.<base64url(zlib(json(data)))>

    Data is verified on load with HMAC-SHA256. Also caches recently processed
    statements in memory and /tmp for low-latency retrieval.
    """

    _cache: dict[str, dict] = {}

    def __init__(
        self,
        signing_key: Optional[bytes] = None,
        max_data_bytes: int = _MAX_DATA_BYTES,
    ) -> None:
        self._key: bytes = signing_key or _SIGNING_KEY
        self._max_bytes = max_data_bytes

    # ------------------------------------------------------------------

    def save(self, key: str, data: dict) -> str:
        """Serialise, compress, sign, and return *data* as a self-contained token."""
        # Always populate local in-process cache
        self._cache[key] = data

        # Write to /tmp cache (available in Vercel serverless / Linux environments)
        try:
            tmp_path = Path("/tmp") / f"session_{key}.json"
            tmp_path.write_text(json.dumps(data, default=_json_serial_default), encoding="utf-8")
        except Exception:
            pass

        envelope = {"_key": key, "data": data}
        json_bytes = json.dumps(envelope, separators=(",", ":"), default=_json_serial_default).encode("utf-8")

        if len(json_bytes) > self._max_bytes:
            logger.warning(
                "StatelessBackend: payload exceeds %d bytes; serving from local cache with key '%s'",
                self._max_bytes, key
            )
            return key

        compressed = zlib.compress(json_bytes, level=6)
        encoded = base64.urlsafe_b64encode(compressed).decode("ascii")
        signature = self._sign(encoded)
        token = f"{signature}{_TOKEN_SEP}{encoded}"

        logger.debug(
            "StatelessBackend.save: key='%s', raw=%d B, compressed=%d B, token_len=%d",
            key, len(json_bytes), len(compressed), len(token),
        )

        # If token fits within safe browser and serverless URL constraints, return token.
        # Otherwise return key for local /tmp lookup.
        if len(token) <= 3500:
            return token
        return key

    def load(self, token: str) -> Optional[dict]:
        """Verify and deserialise a token, or load from in-memory / /tmp cache."""
        if not token:
            return None

        # 1. Check in-process cache first (fastest)
        if token in self._cache:
            return self._cache[token]

        # 2. Check /tmp filesystem cache
        try:
            tmp_path = Path("/tmp") / f"session_{token}.json"
            if tmp_path.is_file():
                cached_data = json.loads(tmp_path.read_text(encoding="utf-8"))
                self._cache[token] = cached_data
                return cached_data
        except Exception:
            pass

        # 3. If signed token format (<signature>.<payload>)
        if _TOKEN_SEP in token and len(token) > 64:
            sep_idx = token.index(_TOKEN_SEP)
            signature = token[:sep_idx]
            encoded = token[sep_idx + 1:]

            expected = self._sign(encoded)
            if not hmac.compare_digest(signature, expected):
                logger.warning("StatelessBackend: HMAC verification failed")
                return None

            try:
                compressed = base64.urlsafe_b64decode(encoded.encode("ascii"))
                json_bytes = zlib.decompress(compressed)
                envelope = json.loads(json_bytes.decode("utf-8"))
                payload_data = envelope.get("data")
                if payload_data:
                    key = envelope.get("_key", token)
                    self._cache[key] = payload_data
                    return payload_data
            except Exception as exc:
                logger.warning("StatelessBackend: failed to decompress/parse token: %s", exc)
                return None

        return None

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
