"""Normalizers for dates, amounts, descriptions, and OCR text cleanup.

Provides pure-function normalizers used across all bank-profile parsers.
No I/O side-effects; all functions are stateless and thread-safe.
"""
from __future__ import annotations

import re
import logging
import unicodedata
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from datetime import date, datetime
from typing import Optional, Tuple

from dateutil import parser as dateutil_parser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MONTH_MAP: dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Regex for DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY (and YY variants)
_DMY_SLASH = re.compile(
    r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2}|\d{4})$"
)
# ISO date YYYY-MM-DD
_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
# DD MMM YYYY  or  DD MMM YY  or  DD MMM
_DD_MON_YYYY = re.compile(
    r"^(\d{1,2})\s+([A-Za-z]{3,9})(?:\s+(\d{2}|\d{4}))?$"
)
# MMM DD, YYYY  or  MMM DD  (US-style)
_MON_DD_YYYY = re.compile(
    r"^([A-Za-z]{3,9})\s+(\d{1,2}),?(?:\s+(\d{2}|\d{4}))?$"
)

# Amount patterns
_BRACKETED = re.compile(r"^\(([^)]+)\)$")
_DR_CR = re.compile(r"(?:(?<=\d)|(?<=\s)|(?<=^))(Dr|CR|Cr|DR)\b", re.IGNORECASE)
_SIGN_PREFIX = re.compile(r"^([+\-])")
_SPACE_IN_NUMBER = re.compile(r"(\d)\s+(\d)")
_CURRENCY_SYMBOL = re.compile(r"[₹$€£¥]")

# Description noise patterns
_MULTI_SPACE = re.compile(r"\s{2,}")
_REF_PATTERN = re.compile(
    r"\b(?:ref(?:erence)?|txn|trans(?:action)?|id|no\.?)\s*[:\s]\s*[\w/\-]+",
    re.IGNORECASE,
)

# Payment channel detection patterns (order matters – most specific first)
_CHANNEL_PATTERNS: list[Tuple[str, re.Pattern[str]]] = [
    ("UPI",      re.compile(r"UPI[/\-]|[/\-]UPI\b", re.IGNORECASE)),
    ("NEFT",     re.compile(r"\bNEFT\b", re.IGNORECASE)),
    ("IMPS",     re.compile(r"\bIMPS\b", re.IGNORECASE)),
    ("RTGS",     re.compile(r"\bRTGS\b", re.IGNORECASE)),
    ("ATM",      re.compile(r"^ATM\b|ATM[\s/]WDL|ATM/WDL", re.IGNORECASE)),
    ("POS",      re.compile(r"\bPOS\b|/POS/|\bPURCHASE\b", re.IGNORECASE)),
    ("CHQ",      re.compile(r"\bCLG/|\bCHQ\b|\bCHEQUE\b", re.IGNORECASE)),
    ("ACH",      re.compile(r"\bACH\b|\bECS\b|\bNACH\b", re.IGNORECASE)),
    ("CASH",     re.compile(r"CASH\s+DEPOSIT|CASH\s+WDL", re.IGNORECASE)),
    ("INTEREST", re.compile(r"\bINT(?:EREST)?\b", re.IGNORECASE)),
    ("CHARGES",  re.compile(r"\bCHRG\b|\bCHARGE[S]?\b|\bFEE[S]?\b", re.IGNORECASE)),
]

# Noise words to strip when extracting counterparty
_NOISE_WORDS: set[str] = {
    "upi", "neft", "imps", "rtgs", "atm", "pos", "chq", "ach", "nach", "ecs",
    "cash", "interest", "charges", "transfer", "payment", "txn", "ref",
    "dr", "cr", "debit", "credit", "bank", "ltd", "pvt", "india",
}
_COUNTERPARTY_REF = re.compile(r"\b\d{6,}\b")
_COUNTERPARTY_DATE = re.compile(
    r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b|\b\d{1,2}\s+[A-Za-z]{3}\s+\d{2,4}\b"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expand_year(year_2d: int, hint: Optional[int] = None) -> int:
    """Expand a 2-digit year to 4-digit using sliding window heuristic."""
    if year_2d <= 29:
        return 2000 + year_2d
    if year_2d <= 99:
        return 1900 + year_2d
    return year_2d  # already 4-digit (shouldn't happen in this branch)


def _safe_date(year: int, month: int, day: int) -> Optional[date]:
    """Construct a date, returning None on ValueError."""
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _mask_account_info(text: str) -> str:
    """Replace digits in account-like sequences with X for safe logging."""
    return re.sub(r"\d{4,}", lambda m: "X" * len(m.group()), text)


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

def normalize_date(raw: str, year_hint: Optional[int] = None) -> Optional[date]:
    """Parse a raw date string into a :class:`datetime.date`.

    Supported formats
    -----------------
    - ``DD/MM/YYYY``, ``DD/MM/YY``
    - ``DD-MM-YYYY``, ``DD-MM-YY``
    - ``DD.MM.YYYY``, ``DD.MM.YY``
    - ``DD MMM YYYY``, ``DD MMM YY``, ``DD MMM`` (year from *year_hint* or current year)
    - ``MMM DD, YYYY`` (US style)
    - ``YYYY-MM-DD`` (ISO 8601)
    - Falls back to :mod:`dateutil.parser`

    Parameters
    ----------
    raw:
        Raw date string extracted from a bank statement.
    year_hint:
        Year to use when no year is present in *raw* (e.g. ``2025``).

    Returns
    -------
    :class:`datetime.date` or ``None`` if the string cannot be parsed.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()
    fallback_year = year_hint or datetime.now().year

    # --- ISO YYYY-MM-DD ---
    m = _ISO_DATE.match(text)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # --- DD/MM/YYYY or DD-MM-YYYY ---
    m = _DMY_SLASH.match(text)
    if m:
        day, month, year_raw = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if len(m.group(3)) == 2:
            year_raw = _expand_year(year_raw, year_hint)
        return _safe_date(year_raw, month, day)

    # --- DD MMM YYYY / DD MMM YY / DD MMM ---
    m = _DD_MON_YYYY.match(text)
    if m:
        day = int(m.group(1))
        month_str = m.group(2).lower()
        month = _MONTH_MAP.get(month_str)
        if month is None:
            logger.warning("Unknown month abbreviation in date: %s", _mask_account_info(text))
            return None
        year_raw = int(m.group(3)) if m.group(3) else fallback_year
        if year_raw < 100:
            year_raw = _expand_year(year_raw, year_hint)
        return _safe_date(year_raw, month, day)

    # --- MMM DD, YYYY / MMM DD ---
    m = _MON_DD_YYYY.match(text)
    if m:
        month_str = m.group(1).lower()
        month = _MONTH_MAP.get(month_str)
        if month is None:
            return None
        day = int(m.group(2))
        year_raw = int(m.group(3)) if m.group(3) else fallback_year
        if year_raw < 100:
            year_raw = _expand_year(year_raw, year_hint)
        return _safe_date(year_raw, month, day)

    # --- dateutil fallback ---
    try:
        parsed = dateutil_parser.parse(text, dayfirst=True, default=datetime(fallback_year, 1, 1))
        logger.debug("dateutil parsed '%s' → %s", _mask_account_info(text), parsed.date())
        return parsed.date()
    except (ValueError, OverflowError):
        logger.warning("Could not parse date: '%s'", _mask_account_info(text))
        return None


# ---------------------------------------------------------------------------
# Amount normalization
# ---------------------------------------------------------------------------

def normalize_amount(raw: str) -> Optional[Decimal]:
    """Parse a raw amount string into a :class:`decimal.Decimal`.

    Rules applied (in order)
    ------------------------
    1. Empty / nil values → ``Decimal("0")``
    2. Bracketed negatives ``(1,250.00)`` → ``-1250.00``
    3. Dr/Cr suffix stripped (direction is handled by :func:`extract_amount_and_direction`)
    4. Currency symbols stripped
    5. Stray spaces inside number collapsed
    6. Indian comma grouping (e.g. ``1,23,456.78``) normalised
    7. Western comma grouping normalised
    8. Final :class:`~decimal.Decimal` parse; ``None`` returned on failure

    Parameters
    ----------
    raw:
        Raw amount string from parser (e.g. ``"1,23,456.78 Dr"``).

    Returns
    -------
    :class:`~decimal.Decimal` or ``None`` if truly unparseable.
    """
    if not raw:
        return Decimal("0")

    text = raw.strip()

    # Nil / dash shorthand
    if text in ("-", "--", "NIL", "nil", "N/A", "n/a", ""):
        return Decimal("0")

    # Stray spaces inside the number (e.g. "1 250.00")
    text = _SPACE_IN_NUMBER.sub(r"\1\2", text)

    # Strip currency symbols
    text = _CURRENCY_SYMBOL.sub("", text).strip()

    # Detect and strip Dr/Cr suffix (keep sign info only for caller)
    text = _DR_CR.sub("", text).strip()

    # Detect sign prefix
    negative = False
    if _BRACKETED.match(text):
        negative = True
        text = _BRACKETED.match(text).group(1).strip()  # type: ignore[union-attr]
    elif text.startswith("-"):
        negative = True
        text = text[1:].strip()
    elif text.startswith("+"):
        text = text[1:].strip()

    # Remove thousands commas (handle both Indian and Western groupings)
    # First remove all commas; Decimal will handle the period as decimal separator
    text = text.replace(",", "")

    # Handle multiple periods (e.g. "118.750.00" or "1.18.341.72" from OCR reading commas as dots)
    if text.count(".") > 1:
        parts = text.split(".")
        text = "".join(parts[:-1]) + "." + parts[-1]

    if not text:
        return Decimal("0")

    try:
        value = Decimal(text)
        if abs(value) > Decimal("999999999999"):
            return None
    except (InvalidOperation, ValueError):
        logger.warning("Could not parse amount: '%s'", raw)
        return None

    if negative:
        value = -value

    return value


def extract_amount_and_direction(raw: str) -> Tuple[Optional[Decimal], Optional[str]]:
    """Extract an amount and its debit/credit direction from a raw string.

    Parameters
    ----------
    raw:
        Raw field value such as ``"1,250.00 Dr"`` or ``"(500.00)"`` or ``"CR 1250"``.

    Returns
    -------
    ``(amount, direction)`` where *direction* is ``"DR"``, ``"CR"``, or ``None``.
    The returned *amount* is always the **absolute value** when direction is
    encoded via Dr/Cr suffix or bracket; it may be negative if a bare ``-``
    sign prefix is present and no direction keyword was found.
    """
    if not raw:
        return Decimal("0"), None

    text = raw.strip()

    direction: Optional[str] = None

    # Check for explicit Dr/Cr suffix / prefix
    dr_cr_match = _DR_CR.search(text)
    if dr_cr_match:
        kw = dr_cr_match.group(1).upper()
        direction = "DR" if kw == "DR" else "CR"

    # Bracketed → debit
    if _BRACKETED.match(text):
        direction = direction or "DR"

    amount = normalize_amount(text)

    # Ensure amount is positive when direction is explicit
    if direction is not None and amount is not None and amount < 0:
        amount = -amount

    return amount, direction


# ---------------------------------------------------------------------------
# OCR cleanup
# ---------------------------------------------------------------------------

# Map of OCR misread characters to correct digits (used in numeric contexts)
_OCR_DIGIT_MAP: dict[str, str] = {
    "O": "0", "o": "0",
    "l": "1", "I": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
}

_OCR_DIGIT_RE = re.compile(r"[OolISBZG]")
_OCR_CURRENCY_FIXES: list[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"Rs\.?\s*"), "₹"),
    (re.compile(r"INR\s*"), "₹"),
]
# Stray comma between non-digit or inside a word
_STRAY_COMMA = re.compile(r"(?<!\d),(?!\d)|(?<=\d),(?=[^\d\s])")


def clean_ocr_text(text: str) -> str:
    """Apply light OCR post-processing suitable for mixed text/numeric fields.

    Character substitutions are applied only to tokens that look numeric
    (contain at least one digit) so that alphabetic description text is
    preserved.

    Parameters
    ----------
    text:
        Raw OCR output containing mixed text.

    Returns
    -------
    Cleaned string with common OCR errors corrected.
    """
    if not text:
        return text

    # Fix common currency symbol misreads
    for pattern, replacement in _OCR_CURRENCY_FIXES:
        text = pattern.sub(replacement, text)

    # Remove stray commas not between digits
    text = _STRAY_COMMA.sub("", text)

    # Apply digit substitutions only within numeric-looking tokens
    tokens = text.split()
    cleaned_tokens: list[str] = []
    for token in tokens:
        if re.search(r"\d", token):
            token = _OCR_DIGIT_RE.sub(lambda m: _OCR_DIGIT_MAP.get(m.group(), m.group()), token)
        cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)


def clean_numeric_ocr(text: str) -> str:
    """Aggressive OCR cleanup for fields that should contain only numbers.

    Applies full character substitution map, then strips everything that is
    not a digit, comma, period, or leading dash.

    Parameters
    ----------
    text:
        Raw OCR output from an amount or balance field.

    Returns
    -------
    Cleaned numeric string ready for :func:`normalize_amount`.
    """
    if not text:
        return text

    # Apply all OCR substitutions
    result = _OCR_DIGIT_RE.sub(lambda m: _OCR_DIGIT_MAP.get(m.group(), m.group()), text)

    # Strip everything that is not numeric, comma, period, or leading dash
    result = re.sub(r"[^\d,.\-]", "", result)

    # Remove any dash that isn't at the start
    result = re.sub(r"(?<!^)-", "", result)

    return result.strip()


# ---------------------------------------------------------------------------
# Description normalization
# ---------------------------------------------------------------------------

def normalize_description(raw: str) -> str:
    """Clean a transaction description string.

    Collapses whitespace, strips leading/trailing whitespace, and applies
    Unicode NFKD normalization. Reference numbers are **not** removed.

    Parameters
    ----------
    raw:
        Raw description string from the parsed row.

    Returns
    -------
    Normalised description string.
    """
    if not raw:
        return ""

    # Unicode normalization (NFKD) then re-encode to ASCII-compatible form,
    # but keep non-ASCII characters that survive decomposition
    normalized = unicodedata.normalize("NFKD", raw)

    # Collapse multiple whitespace characters (including \t, \n) to a single space
    normalized = _MULTI_SPACE.sub(" ", normalized)

    return normalized.strip()


def extract_payment_channel(description: str) -> str:
    """Identify the payment channel from a transaction description.

    Parameters
    ----------
    description:
        Normalised transaction description string.

    Returns
    -------
    One of: ``"UPI"``, ``"NEFT"``, ``"IMPS"``, ``"RTGS"``, ``"ATM"``,
    ``"POS"``, ``"CHQ"``, ``"ACH"``, ``"CASH"``, ``"INTEREST"``,
    ``"CHARGES"``, or ``"OTHER"``.
    """
    if not description:
        return "OTHER"

    for channel, pattern in _CHANNEL_PATTERNS:
        if pattern.search(description):
            logger.debug("Payment channel '%s' detected in: %.60s", channel, description)
            return channel

    return "OTHER"


def extract_counterparty(description: str, payment_channel: str) -> str:
    """Extract the merchant or counterparty name from a transaction description.

    Strips the payment channel prefix, reference numbers, embedded dates,
    and known noise words to arrive at the most likely counterparty name.

    Parameters
    ----------
    description:
        Normalised transaction description.
    payment_channel:
        Channel string as returned by :func:`extract_payment_channel`.

    Returns
    -------
    Best-effort counterparty/merchant name, or empty string if extraction fails.
    """
    if not description:
        return ""

    text = description.strip()

    # Remove embedded dates
    text = _COUNTERPARTY_DATE.sub("", text)

    # Remove long reference/transaction IDs (6+ digit runs)
    text = _COUNTERPARTY_REF.sub("", text)

    # Tokenize and filter noise words / channel keyword
    tokens = re.split(r"[\s/\-|:]+", text)
    cleaned: list[str] = []
    for token in tokens:
        low = token.lower().strip(".,")
        if not low:
            continue
        if low in _NOISE_WORDS:
            continue
        if low == payment_channel.lower():
            continue
        cleaned.append(token.strip(".,"))

    result = " ".join(cleaned).strip()

    # Collapse any remaining multiple spaces
    result = _MULTI_SPACE.sub(" ", result)

    return result


# ---------------------------------------------------------------------------
# Number formatting helpers
# ---------------------------------------------------------------------------

def format_indian_number(value: Decimal) -> str:
    """Format a :class:`~decimal.Decimal` using the Indian numbering system.

    Example: ``Decimal("1234567.89")`` → ``"12,34,567.89"``

    Parameters
    ----------
    value:
        Decimal value to format.

    Returns
    -------
    String formatted with Indian grouping (last 3 digits, then groups of 2).
    """
    # Round to 2 decimal places
    value = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    negative = value < 0
    abs_val = abs(value)

    # Split integer and fractional parts
    str_val = f"{abs_val:.2f}"
    int_part, frac_part = str_val.split(".")

    # Indian grouping: last 3 digits, then groups of 2
    if len(int_part) <= 3:
        formatted_int = int_part
    else:
        # Last 3 digits
        last3 = int_part[-3:]
        rest = int_part[:-3]
        # Group rest in 2s from the right
        groups: list[str] = []
        while rest:
            groups.append(rest[-2:])
            rest = rest[:-2]
        groups.reverse()
        formatted_int = ",".join(groups) + "," + last3

    result = f"{formatted_int}.{frac_part}"
    return f"-{result}" if negative else result


def is_zero_amount(value: Optional[Decimal]) -> bool:
    """Return ``True`` if *value* is ``None``, zero, or negligibly small.

    Parameters
    ----------
    value:
        Amount to check.

    Returns
    -------
    ``True`` when the amount is effectively zero (absolute value < 0.001).
    """
    if value is None:
        return True
    return abs(value) < Decimal("0.001")


class StandardRowNormalizer:
    """Row normalizer converting raw table cell strings to typed fields."""

    def normalize(self, raw: Any, year_hint: int = 0) -> dict[str, Any]:
        """Normalize a single raw extracted row."""
        flags: list[str] = []

        def _val(key: str, default: Any = "") -> Any:
            if isinstance(raw, dict):
                return raw.get(key, default)
            return getattr(raw, key, default)

        # 1. Date normalization
        norm_date = normalize_date(_val("date_raw", ""), year_hint=year_hint or None)
        date_str = norm_date.isoformat() if norm_date else None
        if not date_str and str(_val("date_raw", "")).strip():
            flags.append(f"Unparseable date: '{_val('date_raw')}'")

        # 2. Value date
        norm_val_date = normalize_date(_val("value_date_raw", ""), year_hint=year_hint or None)
        val_date_str = norm_val_date.isoformat() if norm_val_date else None

        # 3. Description
        desc_clean = normalize_description(_val("description", ""))

        # 4. Ref No
        ref_clean = str(_val("ref_no", "")).strip()

        # 5. Amounts
        raw_dr = str(_val("debit_raw", "")).strip()
        raw_cr = str(_val("credit_raw", "")).strip()
        raw_bal = str(_val("balance_raw", "")).strip()

        debit_dec = normalize_amount(raw_dr) if raw_dr and raw_dr not in ("-", "--", "N/A", "n/a", "NIL", "nil") else None
        credit_dec = normalize_amount(raw_cr) if raw_cr and raw_cr not in ("-", "--", "N/A", "n/a", "NIL", "nil") else None
        balance_dec = normalize_amount(raw_bal) if raw_bal and raw_bal not in ("-", "--", "N/A", "n/a", "NIL", "nil") else None

        # Check if raw_dr or raw_cr actually had a Dr/Cr suffix embedded
        if raw_dr and not raw_cr:
            _, d_dir = extract_amount_and_direction(raw_dr)
            if d_dir == "CR":
                credit_dec, debit_dec = debit_dec, None
        elif raw_cr and not raw_dr:
            _, d_dir = extract_amount_and_direction(raw_cr)
            if d_dir == "DR":
                debit_dec, credit_dec = credit_dec, None

        # If layout used single amount column
        amt_raw = _val("amount_raw", "")
        if amt_raw and debit_dec is None and credit_dec is None:
            parsed_amt, d_dir = extract_amount_and_direction(amt_raw)
            dr_cr_val = _val("dr_cr_raw", "") or d_dir or ""
            if "DR" in dr_cr_val.upper():
                debit_dec = abs(parsed_amt) if parsed_amt is not None else None
            elif "CR" in dr_cr_val.upper():
                credit_dec = abs(parsed_amt) if parsed_amt is not None else None
            elif parsed_amt is not None and parsed_amt < 0:
                debit_dec = abs(parsed_amt)
            elif parsed_amt is not None and parsed_amt > 0:
                credit_dec = parsed_amt

        # Handle negative values in debit or credit columns (e.g. -2889.28 in credit is actually a debit)
        if credit_dec is not None and credit_dec < 0 and debit_dec is None:
            debit_dec = abs(credit_dec)
            credit_dec = None
        elif debit_dec is not None and debit_dec < 0:
            debit_dec = abs(debit_dec)

        # Recombine split decimal amount if integer and cents were split across credit/debit
        if debit_dec is not None and credit_dec is not None:
            raw_dr = str(_val("debit_raw", "")).strip()
            raw_cr = str(_val("credit_raw", "")).strip()
            if credit_dec >= 100 and debit_dec < 100 and debit_dec > 0:
                if raw_cr.endswith(".") or "dr" in raw_dr.lower() or "cr" in raw_dr.lower():
                    combined = credit_dec + (debit_dec / Decimal("100"))
                    if "CR" in raw_dr.upper() or "CR" in raw_cr.upper():
                        credit_dec = combined
                        debit_dec = None
                    else:
                        debit_dec = combined
                        credit_dec = None
            elif debit_dec >= 100 and credit_dec < 100 and credit_dec > 0:
                if raw_dr.endswith(".") or "dr" in raw_cr.lower() or "cr" in raw_cr.lower():
                    combined = debit_dec + (credit_dec / Decimal("100"))
                    if "CR" in raw_cr.upper() or "CR" in raw_dr.upper():
                        credit_dec = combined
                        debit_dec = None
                    else:
                        debit_dec = combined
                        credit_dec = None

        debit_val = float(debit_dec) if debit_dec is not None and not is_zero_amount(debit_dec) else None
        credit_val = float(credit_dec) if credit_dec is not None and not is_zero_amount(credit_dec) else None
        balance_val = float(balance_dec) if balance_dec is not None else None

        return {
            "date": date_str,
            "value_date": val_date_str,
            "description": desc_clean,
            "ref_no": ref_clean,
            "debit": debit_val,
            "credit": credit_val,
            "balance": balance_val,
            "page_num": _val("page_num", 1),
            "flag_reasons": flags,
        }

