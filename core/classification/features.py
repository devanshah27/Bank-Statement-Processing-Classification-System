"""Feature extraction for the classification engine.

Parses raw transaction description strings into a rich set of structured
features consumed by both the rules engine and the ML model.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regular expressions (module-level for performance)
# ---------------------------------------------------------------------------

# Long numeric sequences that are reference / UTR / IMPS numbers (≥ 9 digits)
_RE_LONG_NUMERIC = re.compile(r"\b\d{9,}\b")

# Common UPI reference patterns: "UPI/", "UPI-", "upi ref", etc.
_RE_UPI_REF = re.compile(
    r"\bupi[\s/\-]*(?:ref(?:erence)?|txn|trans(?:action)?)?[\s:]*\d{6,}\b",
    re.IGNORECASE,
)

# IMPS / NEFT / RTGS reference numbers
_RE_IMPS_REF = re.compile(
    r"\b(?:imps|neft|rtgs)[\s/\-]*(?:ref(?:no)?)?[\s:]*\d{6,}\b",
    re.IGNORECASE,
)

# Date-like patterns embedded in descriptions (DD-MM-YYYY, DD/MM/YY, etc.)
_RE_DATE = re.compile(
    r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{2}[-/]\d{2})\b"
)

# Amount-like patterns embedded in descriptions (e.g. "INR 1,500.00", "Rs.500")
_RE_AMOUNT = re.compile(
    r"(?:inr|rs\.?|₹)\s*[\d,]+(?:\.\d{1,2})?", re.IGNORECASE
)

# Account number patterns (10–18 digit strings often formatted with Xs)
_RE_ACCOUNT_NUMBER = re.compile(
    r"\b(?:[xX*]{2,}\d{3,}|\d{3,}[xX*]{2,}|\b\d{10,18}\b)"
)

# Payment channel keywords mapped to canonical names
_CHANNEL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("UPI",       re.compile(r"\bupi\b", re.IGNORECASE)),
    ("NEFT",      re.compile(r"\bneft\b", re.IGNORECASE)),
    ("RTGS",      re.compile(r"\brtgs\b", re.IGNORECASE)),
    ("IMPS",      re.compile(r"\bimps\b", re.IGNORECASE)),
    ("ATM",       re.compile(r"\batm\b", re.IGNORECASE)),
    ("POS",       re.compile(r"\bpos\b|\bpoint[\s\-]of[\s\-]sale\b", re.IGNORECASE)),
    ("CHQ",       re.compile(r"\bch(?:eq(?:ue)?|q)\b|\bcheque\b|\bcheck\b", re.IGNORECASE)),
    ("ACH",       re.compile(r"\bach\b|\bein\b|\bnach\b|\becs\b", re.IGNORECASE)),
    ("CASH",      re.compile(r"\bcash\b|\bdeposit\b", re.IGNORECASE)),
    ("INTEREST",  re.compile(r"\binterest\b|\bint(?:\s|$)", re.IGNORECASE)),
    ("CHARGES",   re.compile(
        r"\bcharge[sd]?\b|\bfee\b|\bpenalt(?:y|ies)\b|\bgst\b|\btax\b|\bservice\s+charge\b",
        re.IGNORECASE,
    )),
]

# Noise words to strip when extracting counterparty
_RE_NOISE = re.compile(
    r"\b(?:transfer|payment|paid|received|from|to|via|through|on|by|for|the|a|an)\b",
    re.IGNORECASE,
)

# Tokenisation splitter
_RE_SPLIT = re.compile(r"[^a-z0-9]+")

# Amount band boundaries
_BAND_MICRO  = Decimal("500")
_BAND_SMALL  = Decimal("5000")
_BAND_MEDIUM = Decimal("50000")
_BAND_LARGE  = Decimal("500000")


# ---------------------------------------------------------------------------
# Data-class
# ---------------------------------------------------------------------------

@dataclass
class TransactionFeatures:
    """Rich feature set extracted from a single transaction description."""

    description_cleaned: str        # lower-cased, refs and amounts stripped
    description_normalized: str     # payment-channel keywords normalised
    payment_channel: str            # UPI/NEFT/IMPS/RTGS/ATM/POS/CHQ/ACH/CASH/INTEREST/CHARGES/OTHER
    direction: str                  # DR or CR
    amount: Decimal
    amount_band: str                # micro/small/medium/large/xlarge
    counterparty: str               # extracted merchant/person name
    has_utr: bool
    has_account_number: bool
    is_round_amount: bool


def denoise_ocr(text: str) -> str:
    """Correct common OCR digit substitutions (0->o, 1->i, 5->s) in alphabetic tokens."""
    tokens = text.split()
    out = []
    for tok in tokens:
        if "1mg" in tok.lower():
            out.append(tok)
            continue
        if re.search(r"[a-zA-Z]", tok) and not re.match(r"^[a-zA-Z]+[0-9]{4,}$", tok):
            t_clean = tok.replace("0", "o").replace("1", "i").replace("5", "s")
            out.append(t_clean)
        else:
            out.append(tok)
    return " ".join(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    description: str,
    direction: str,
    amount: Optional[Decimal] = None,
) -> TransactionFeatures:
    """Extract structured classification features from a transaction description.

    Parameters
    ----------
    description:
        Raw transaction narration / description string.
    direction:
        ``"DR"`` for debit or ``"CR"`` for credit (case-insensitive).
    amount:
        Transaction amount as :class:`~decimal.Decimal`, or ``None``.

    Returns
    -------
    TransactionFeatures
        Fully populated feature object ready for rule-matching or ML inference.
    """
    if not description:
        logger.debug("extract_features called with empty description")
        description = ""

    safe_amount = amount if amount is not None else Decimal("0")
    direction_raw = direction.upper().strip() if direction else "DR"
    if direction_raw in ("DEBIT", "DR"):
        direction_norm = "DR"
    elif direction_raw in ("CREDIT", "CR"):
        direction_norm = "CR"
    else:
        direction_norm = "DR"

    # Apply OCR de-noising to raw description
    denoised_desc = denoise_ocr(description)

    # --- UTR / account-number detection (on raw description) ---
    has_utr = bool(
        _RE_LONG_NUMERIC.search(description)
        or _RE_UPI_REF.search(description)
        or _RE_IMPS_REF.search(description)
    )
    has_account_number = bool(_RE_ACCOUNT_NUMBER.search(description))

    # --- Clean: lowercase → strip refs, dates, amounts ---
    cleaned = denoised_desc.lower()
    cleaned = _RE_UPI_REF.sub(" ", cleaned)
    cleaned = _RE_IMPS_REF.sub(" ", cleaned)
    cleaned = _RE_LONG_NUMERIC.sub(" ", cleaned)
    cleaned = _RE_DATE.sub(" ", cleaned)
    cleaned = _RE_AMOUNT.sub(" ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    # --- Payment channel detection ---
    payment_channel = "OTHER"
    for channel_name, pattern in _CHANNEL_PATTERNS:
        if pattern.search(denoised_desc):
            payment_channel = channel_name
            break

    # --- Normalise description: replace channel keyword with canonical tag ---
    normalised = cleaned
    for channel_name, pattern in _CHANNEL_PATTERNS:
        normalised = pattern.sub(f"__{channel_name}__", normalised)
    normalised = re.sub(r"\s{2,}", " ", normalised).strip()

    # --- Counterparty extraction ---
    # Start from cleaned description; remove channel noise words, strip punctuation
    cp_text = cleaned
    # Remove channel keywords
    for _, pattern in _CHANNEL_PATTERNS:
        cp_text = pattern.sub(" ", cp_text)
    # Remove noise words
    cp_text = _RE_NOISE.sub(" ", cp_text)
    # Remove account-number-like strings
    cp_text = _RE_ACCOUNT_NUMBER.sub(" ", cp_text)
    # Collapse whitespace and strip punctuation
    cp_text = re.sub(r"[^a-z0-9 ]", " ", cp_text)
    cp_text = re.sub(r"\s{2,}", " ", cp_text).strip()
    counterparty = cp_text[:60]  # cap length

    # --- Amount band ---
    amount_band = _get_amount_band(safe_amount)

    # --- Round amount ---
    try:
        is_round_amount = (safe_amount % Decimal("100") == Decimal("0")) and (
            safe_amount > Decimal("0")
        )
    except Exception:
        is_round_amount = False

    return TransactionFeatures(
        description_cleaned=cleaned,
        description_normalized=normalised,
        payment_channel=payment_channel,
        direction=direction_norm,
        amount=safe_amount,
        amount_band=amount_band,
        counterparty=counterparty,
        has_utr=has_utr,
        has_account_number=has_account_number,
        is_round_amount=is_round_amount,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_amount_band(amount: Decimal) -> str:
    """Classify *amount* into a named band.

    Bands
    -----
    * ``micro``  – < 500
    * ``small``  – 500 ≤ x < 5 000
    * ``medium`` – 5 000 ≤ x < 50 000
    * ``large``  – 50 000 ≤ x < 500 000
    * ``xlarge`` – ≥ 500 000
    """
    if amount < _BAND_MICRO:
        return "micro"
    if amount < _BAND_SMALL:
        return "small"
    if amount < _BAND_MEDIUM:
        return "medium"
    if amount < _BAND_LARGE:
        return "large"
    return "xlarge"
