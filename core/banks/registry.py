"""Bank profile registry with auto-detection from PDF text.

Loads bank profiles from YAML files in the ``profiles/`` sub-directory and
exposes an auto-detection API so callers never need to hard-code bank names.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).parent / "profiles"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class BankProfile:
    """Immutable descriptor for a single bank's statement format.

    All attributes map 1-to-1 with the YAML profile files stored in
    ``core/banks/profiles/``.
    """

    name: str
    """Canonical short name, e.g. ``"sbi"`` or ``"hdfc"``."""

    display_name: str
    """Human-readable name shown in the UI."""

    ifsc_prefix: list[str] = field(default_factory=list)
    """IFSC code prefixes that identify this bank, e.g. ``["SBIN", "SBIH"]``."""

    header_keywords: list[str] = field(default_factory=list)
    """Phrases searched (case-insensitively) in page headers to identify the bank."""

    date_formats: list[str] = field(default_factory=list)
    """``strptime`` format strings tried in order when parsing dates."""

    amount_format: str = "indian"
    """``"indian"`` (1,00,000.00) or ``"western"`` (100,000.00)."""

    decimal_separator: str = "."
    """Decimal separator character: ``"."`` or ``","``."""

    debit_credit_style: str = "separate"
    """Layout variant:

    * ``"separate"``          – dedicated Debit and Credit columns
    * ``"single_with_suffix"`` – one Amount column plus a Dr/Cr column
    * ``"signed"``            – negative = debit, positive = credit
    * ``"bracketed"``         – ``(1250.00)`` = debit
    """

    column_order_hint: list[str] = field(default_factory=list)
    """Expected column order (advisory, not enforced)."""

    multiline_narration: bool = True
    """``True`` if narration text may span multiple PDF rows."""

    value_date_present: bool = False
    """``True`` if the statement contains a separate Value Date column."""

    ref_no_present: bool = False
    """``True`` if the statement contains a Cheque/Ref No. column."""

    skip_rows_containing: list[str] = field(default_factory=list)
    """Exact strings (uppercased) that identify header-repetition / footer rows."""

    extra_patterns: dict = field(default_factory=dict)
    """Bank-specific regex patterns (e.g. narration prefixes, ref-number formats)."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class BankRegistry:
    """Singleton-style registry of :class:`BankProfile` instances.

    Profiles are loaded lazily on first access from the ``profiles/``
    directory.  Additional profiles can be injected at runtime via
    :meth:`register` to support a plug-in pattern.

    Usage::

        registry = BankRegistry()
        profile = registry.detect_bank(first_2000_chars_of_pdf)
    """

    # Minimum combined score (IFSC hits + keyword hits) required to accept a
    # bank detection result instead of falling back to the generic profile.
    _CONFIDENCE_THRESHOLD: float = 0.5

    def __init__(self) -> None:
        self._profiles: dict[str, BankProfile] = {}
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load all profiles if not already done (lazy initialisation)."""
        if not self._loaded:
            self.load_all()

    def load_all(self) -> None:
        """Load every ``*.yaml`` file from :data:`PROFILES_DIR` into the registry.

        Duplicate names are overwritten with a warning so that user-supplied
        profiles (e.g. placed in the same directory) take precedence over
        bundled ones if loaded afterwards.

        Raises
        ------
        FileNotFoundError
            If :data:`PROFILES_DIR` does not exist.
        """
        if not PROFILES_DIR.exists():
            raise FileNotFoundError(
                f"Bank profiles directory not found: {PROFILES_DIR}"
            )

        yaml_files = sorted(PROFILES_DIR.glob("*.yaml"))
        logger.info("Loading %d bank profile(s) from %s", len(yaml_files), PROFILES_DIR)

        for yaml_path in yaml_files:
            try:
                profile = self._load_yaml(yaml_path)
                if profile.name in self._profiles:
                    logger.warning(
                        "Duplicate bank profile '%s' from %s — overwriting",
                        profile.name,
                        yaml_path.name,
                    )
                self._profiles[profile.name] = profile
                logger.debug("Loaded bank profile: %s (%s)", profile.name, yaml_path.name)
            except Exception:
                logger.exception("Failed to load bank profile from %s", yaml_path)

        self._loaded = True
        logger.info("BankRegistry ready with %d profile(s): %s", len(self._profiles), list(self._profiles.keys()))

    @staticmethod
    def _load_yaml(path: Path) -> BankProfile:
        """Parse a single YAML file into a :class:`BankProfile`.

        Parameters
        ----------
        path:
            Absolute path to the ``.yaml`` profile file.

        Returns
        -------
        BankProfile
            Populated dataclass instance.
        """
        with path.open("r", encoding="utf-8") as fh:
            data: dict = yaml.safe_load(fh)

        # Normalise optional fields to their defaults when absent.
        return BankProfile(
            name=str(data["name"]),
            display_name=str(data.get("display_name", data["name"])),
            ifsc_prefix=[str(p) for p in data.get("ifsc_prefix", [])],
            header_keywords=[str(k) for k in data.get("header_keywords", [])],
            date_formats=[str(f) for f in data.get("date_formats", [])],
            amount_format=str(data.get("amount_format", "indian")),
            decimal_separator=str(data.get("decimal_separator", ".")),
            debit_credit_style=str(data.get("debit_credit_style", "separate")),
            column_order_hint=[str(c) for c in data.get("column_order_hint", [])],
            multiline_narration=bool(data.get("multiline_narration", True)),
            value_date_present=bool(data.get("value_date_present", False)),
            ref_no_present=bool(data.get("ref_no_present", False)),
            skip_rows_containing=[str(s) for s in data.get("skip_rows_containing", [])],
            extra_patterns=dict(data.get("extra_patterns", {})),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, profile: BankProfile) -> None:
        """Register a :class:`BankProfile` programmatically (plug-in pattern).

        Parameters
        ----------
        profile:
            The profile to register.  Overwrites any existing profile with the
            same :attr:`~BankProfile.name`.
        """
        self._ensure_loaded()
        if profile.name in self._profiles:
            logger.warning("Overwriting existing profile '%s' via register()", profile.name)
        self._profiles[profile.name] = profile
        logger.info("Registered bank profile: %s", profile.name)

    def get_profile(self, name: str) -> BankProfile:
        """Return the :class:`BankProfile` for *name*.

        Parameters
        ----------
        name:
            The canonical short name, e.g. ``"sbi"``.

        Returns
        -------
        BankProfile

        Raises
        ------
        KeyError
            If no profile with that name is registered.
        """
        self._ensure_loaded()
        if name not in self._profiles:
            raise KeyError(f"No bank profile registered for '{name}'")
        return self._profiles[name]

    def list_banks(self) -> list[str]:
        """Return a sorted list of all registered bank profile names.

        Returns
        -------
        list[str]
            E.g. ``["axis", "generic", "hdfc", "icici", "sbi", ...]``.
        """
        self._ensure_loaded()
        return sorted(self._profiles.keys())

    def detect_bank(self, text: str) -> BankProfile:
        """Auto-detect the bank from the first ~2000 characters of PDF text.

        Detection uses a two-pass scoring approach:

        1. **IFSC prefix scan** – search for IFSC-formatted codes in the text
           and award 2 points per matching prefix.
        2. **Header keyword scan** – case-insensitive search for
           :attr:`~BankProfile.header_keywords`; award 1 point per hit.

        The bank with the highest total score is returned if its normalised
        confidence exceeds :attr:`_CONFIDENCE_THRESHOLD`.  Otherwise the
        ``"generic"`` fallback profile is returned.

        Parameters
        ----------
        text:
            Raw text extracted from the PDF, ideally the first 2000
            characters of the first page or two.

        Returns
        -------
        BankProfile
            Best-matching profile, or ``"generic"`` if none detected.
        """
        self._ensure_loaded()

        sample = text[:2000]
        sample_upper = sample.upper()

        scores: dict[str, float] = {}

        for name, profile in self._profiles.items():
            if name == "generic":
                continue

            score: float = 0.0

            # Pass 1 – IFSC prefix matching (high-confidence signal).
            # IFSC codes are 11-character alphanumeric with the first 4 chars
            # being the bank code.  We look for the prefix appearing at a word
            # boundary followed by digits/letters typical of an IFSC code.
            for prefix in profile.ifsc_prefix:
                pattern = re.compile(
                    r"\b" + re.escape(prefix.upper()) + r"[A-Z0-9]{7}\b",
                    re.IGNORECASE,
                )
                hits = len(pattern.findall(sample_upper))
                if hits:
                    score += 2.0 * min(hits, 3)  # cap to avoid inflating score
                    logger.debug(
                        "detect_bank: %s IFSC prefix '%s' hit %d time(s)",
                        name, prefix, hits,
                    )

            # Pass 2 – Header keyword matching (lower-confidence signal).
            for keyword in profile.header_keywords:
                if keyword.upper() in sample_upper:
                    score += 1.0
                    logger.debug(
                        "detect_bank: %s keyword '%s' found", name, keyword
                    )

            if score > 0:
                scores[name] = score

        if not scores:
            logger.info("detect_bank: no signals found — returning generic profile")
            return self._profiles.get("generic", self._create_fallback_generic())

        best_name = max(scores, key=lambda k: scores[k])
        best_score = scores[best_name]

        # Normalise by maximum achievable score for this profile to get
        # a rough 0–1 confidence estimate.
        best_profile = self._profiles[best_name]
        max_possible = (
            len(best_profile.ifsc_prefix) * 6.0  # 2 pts × cap-of-3
            + len(best_profile.header_keywords) * 1.0
        )
        confidence = best_score / max_possible if max_possible > 0 else 0.0

        logger.info(
            "detect_bank: best='%s' score=%.1f confidence=%.2f",
            best_name, best_score, confidence,
        )

        if confidence < self._CONFIDENCE_THRESHOLD and best_score < 2.0:
            logger.info(
                "detect_bank: confidence %.2f below threshold — returning generic",
                confidence,
            )
            return self._profiles.get("generic", self._create_fallback_generic())

        return self._profiles[best_name]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _create_fallback_generic() -> BankProfile:
        """Return a minimal in-memory generic profile used if the YAML is missing."""
        logger.warning("'generic' profile not found in registry — using hard-coded fallback")
        return BankProfile(
            name="generic",
            display_name="Generic Bank",
            date_formats=["%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d/%m/%y", "%Y-%m-%d"],
            skip_rows_containing=[
                "BROUGHT FORWARD", "CARRIED FORWARD", "BALANCE B/F", "BAL C/F",
                "OPENING BALANCE", "CLOSING BALANCE", "TOTAL", "SUB-TOTAL", "PAGE TOTAL",
            ],
        )

    def __repr__(self) -> str:  # pragma: no cover
        loaded_str = f"loaded={len(self._profiles)}" if self._loaded else "not loaded"
        return f"<BankRegistry {loaded_str}>"
