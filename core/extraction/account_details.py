"""Extract account holder details from bank statement text.

Uses regex patterns, keyword anchors, and positional heuristics to populate
an :class:`AccountDetails` dataclass from raw statement text. No ML or
external APIs are used.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Optional

from core.extraction.normalizers import normalize_date, normalize_amount

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

ACCOUNT_NUMBER_PATTERN = re.compile(
    r"(?:account\s+(?:no\.?|number|num\.?)|a/c\s+no\.?|acc\.?\s+no\.?|a/c)[:\s]+([0-9X]{9,18})",
    re.IGNORECASE,
)

IFSC_PATTERN = re.compile(r"\b([A-Z]{4}0[A-Z0-9]{6})\b")

MICR_PATTERN = re.compile(
    r"(?:micr(?:\s+code)?)[:\s]+(\d{9})",
    re.IGNORECASE,
)

CUSTOMER_ID_PATTERN = re.compile(
    r"(?:customer\s+(?:id|no\.?|number)|cif\s+(?:no\.?|number)|client\s+id)[:\s]+([\w\d\-]+)",
    re.IGNORECASE,
)

BANK_NAME_PATTERN = re.compile(
    r"(?:bank(?:\s+name)?)[:\s]+([\w\s&,\.]+?)(?:\n|$|branch|ifsc)",
    re.IGNORECASE,
)

BRANCH_PATTERN = re.compile(
    r"(?:branch(?:\s+name)?)[:\s]+([\w\s,\.\-]+?)(?:\n|$|ifsc|micr|pin|cif|customer|account|a/c)",
    re.IGNORECASE,
)

ACCOUNT_TYPE_PATTERN = re.compile(
    r"(?:account\s+type|a/c\s+type|acc\s+type)[:\s]+([\w\s]+?)(?:\n|$|account|a/c)",
    re.IGNORECASE,
)

STATEMENT_PERIOD_PATTERN = re.compile(
    r"(?:statement\s+period|period)[:\s]+"
    r"(\d{1,2}[/\-\s][A-Za-z0-9/\-\s]+?)"
    r"\s*(?:to|–|-)\s*"
    r"(\d{1,2}[/\-\s][A-Za-z0-9/\-\s]+?)(?:\n|$|[A-Za-z]{3,})",
    re.IGNORECASE,
)

# "From: 01/04/2024    To: 31/03/2025"
PERIOD_FROM_TO_PATTERN = re.compile(
    r"from[:\s]+(\d{1,2}[/\-\.][A-Za-z0-9\-/\.]+)"
    r"(?:\s+to[:\s]+(\d{1,2}[/\-\.][A-Za-z0-9\-/\.]+))?",
    re.IGNORECASE,
)

OPENING_BALANCE_PATTERN = re.compile(
    r"(?:opening\s*bal(?:ance)?|open(?:ing)?\s*bal\.?|ob)[:\s]+(?:rs\.?|inr|₹)?\s*"
    r"([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{2})?|[0-9]+(?:\.[0-9]{2})?)(?![0-9/])(?:\s*(?:Dr|Cr))?",
    re.IGNORECASE,
)

CLOSING_BALANCE_PATTERN = re.compile(
    r"(?:closing\s*bal(?:ance)?|clos(?:ing)?\s*bal\.?|cb)[:\s]+(?:rs\.?|inr|₹)?\s*"
    r"([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{2})?|[0-9]+(?:\.[0-9]{2})?)(?![0-9/])(?:\s*(?:Dr|Cr))?",
    re.IGNORECASE,
)

CURRENCY_PATTERN = re.compile(
    r"\b(INR|USD|EUR|GBP|AED|SGD)\b",
    re.IGNORECASE,
)

STATEMENT_DATE_PATTERN = re.compile(
    r"(?:statement\s+date|generated\s+on|as\s+of)[:\s]+"
    r"(\d{1,2}[/\-\s][A-Za-z0-9/\-\s]+?)(?:\n|$)",
    re.IGNORECASE,
)

ADDRESS_PATTERN = re.compile(
    r"(?:address|communication\s+address|residence\s+address)[:\s]+(.*?)(?:\n|$|statement\s+period|opening\s*bal|account\s+type|email|mobile|phone)",
    re.IGNORECASE,
)

# Name extraction keyword anchors
_NAME_ANCHORS = re.compile(
    r"(?:account\s+holder|customer\s+name|name\s+of\s+account\s+holder|a/c\s+holder|account\s+name)[:\s]+"
    r"([A-Za-z][A-Za-z\s\.]{2,60}?)(?:\n|$|\d{3}|account|a/c|cif|customer|ifs|micr|branch)",
    re.IGNORECASE,
)

# Known account type keywords for normalisation
_ACCOUNT_TYPE_KEYWORDS: dict[str, str] = {
    "savings": "Savings Account",
    "current": "Current Account",
    "nri": "NRI Account",
    "nro": "NRO Account",
    "nre": "NRE Account",
    "salary": "Salary Account",
    "overdraft": "Overdraft Account",
    "cash credit": "Cash Credit Account",
    "fixed deposit": "Fixed Deposit",
    "recurring deposit": "Recurring Deposit",
}

# Well-known Indian bank name mappings (IFSC prefix → bank name)
_IFSC_BANK_MAP: dict[str, str] = {
    "SBIN": "State Bank of India",
    "HDFC": "HDFC Bank",
    "ICIC": "ICICI Bank",
    "UTIB": "Axis Bank",
    "KKBK": "Kotak Mahindra Bank",
    "PUNB": "Punjab National Bank",
    "BKID": "Bank of India",
    "BARB": "Bank of Baroda",
    "CNRB": "Canara Bank",
    "IOBA": "Indian Overseas Bank",
    "UBIN": "Union Bank of India",
    "INDB": "IndusInd Bank",
    "YESB": "Yes Bank",
    "IDFB": "IDFC First Bank",
    "FDRL": "Federal Bank",
    "KARB": "Karnataka Bank",
    "KVBL": "Karur Vysya Bank",
    "SIBL": "South Indian Bank",
    "CITI": "Citibank",
    "HSBC": "HSBC Bank",
    "SCBL": "Standard Chartered Bank",
    "DEUT": "Deutsche Bank",
    "RBLB": "RBL Bank",
    "BDBL": "Bandhan Bank",
    "FINO": "Fino Payments Bank",
    "AIRP": "Airtel Payments Bank",
    "JAKA": "Jammu & Kashmir Bank",
    "TNSC": "Tamil Nadu State Cooperative Bank",
}


# ---------------------------------------------------------------------------
# AccountDetails dataclass
# ---------------------------------------------------------------------------

@dataclass
class AccountDetails:
    """Structured account metadata extracted from a bank statement.

    All fields default to empty/None so that partial extraction is valid.
    """

    account_holder_name: str = ""
    account_number: str = ""
    ifsc: str = ""
    micr: str = ""
    bank_name: str = ""
    branch: str = ""
    customer_id: str = ""
    account_type: str = ""
    address: Optional[str] = None
    period_from: Optional[date] = None
    period_to: Optional[date] = None
    opening_balance: Optional[Decimal] = None
    closing_balance: Optional[Decimal] = None
    currency: str = "INR"
    statement_date: Optional[date] = None

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def mask_for_log(self) -> str:
        """Return a log-safe representation with sensitive fields masked.

        Account number shows only the last 4 digits; customer ID is fully
        masked.

        Returns
        -------
        str
            Human-readable masked summary suitable for DEBUG log messages.
        """
        masked_acct = (
            "XXXX" + self.account_number[-4:]
            if len(self.account_number) >= 4
            else "XXXX"
        )
        return (
            f"AccountDetails("
            f"name='{self.account_holder_name}', "
            f"acct='{masked_acct}', "
            f"bank='{self.bank_name}', "
            f"ifsc='{self.ifsc}', "
            f"type='{self.account_type}', "
            f"period={self.period_from}→{self.period_to}, "
            f"opening={self.opening_balance}, "
            f"closing={self.closing_balance}, "
            f"currency='{self.currency}'"
            f")"
        )

    def is_complete(self) -> bool:
        """Return ``True`` if the minimum required fields are populated.

        Minimum required: account number, bank name or IFSC.
        """
        return bool(self.account_number and (self.bank_name or self.ifsc))


# ---------------------------------------------------------------------------
# Internal extraction helpers
# ---------------------------------------------------------------------------

def _clean_extracted_name(name: str, bank_name: str = "") -> str:
    noise_prefixes = [
        "statement of account", "bank statement", "account statement",
        "sample national bank", "state bank of india", "hdfc bank", "icici bank", "axis bank",
        "bank", "statement"
    ]
    if bank_name:
        noise_prefixes.insert(0, bank_name.lower())
    changed = True
    while changed:
        changed = False
        low = name.lower().strip()
        for np in noise_prefixes:
            if low.startswith(np):
                name = name.strip()[len(np):].strip(" :-,.")
                changed = True
                break
    return name.strip()


def _extract_name(text: str, bank_name: str = "") -> str:
    """Extract the account holder name using keyword anchors."""
    m = _NAME_ANCHORS.search(text)
    if m:
        raw_match = m.group(1).strip()
        caps_m = re.search(r"\b([A-Z]{3,20}(?:\s+[A-Z]{2,20})+)\b", raw_match)
        if caps_m and not any(w in caps_m.group(1).upper() for w in ["ACCOUNT", "STATEMENT", "BANK", "BRANCH"]):
            return caps_m.group(1).title()
        cleaned = _clean_extracted_name(raw_match, bank_name)
        if len(cleaned) >= 3 and not re.search(r"\d{4,}", cleaned):
            return cleaned.title()

    # Fallback: OCR reading order where name is before 'Account Name' / 'Account Holder'
    rev_m = re.search(r"([A-Za-z\s]{3,60})\s+(?:account\s+name|a/c\s+name|account\s+holder)\b", text, re.IGNORECASE)
    if rev_m:
        raw_match = rev_m.group(1).strip()
        caps_m = re.search(r"\b([A-Z]{3,20}(?:\s+[A-Z]{2,20})+)\b", raw_match)
        if caps_m and not any(w in caps_m.group(1).upper() for w in ["ACCOUNT", "STATEMENT", "BANK", "BRANCH"]):
            return caps_m.group(1).title()
        cleaned = _clean_extracted_name(raw_match, bank_name)
        if len(cleaned) >= 3 and not re.search(r"\d{4,}", cleaned):
            return cleaned.title()

    return ""


def _extract_account_number(text: str) -> str:
    """Extract account number using ACCOUNT_NUMBER_PATTERN."""
    m = ACCOUNT_NUMBER_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    return ""


def _extract_ifsc(text: str) -> str:
    """Extract IFSC code."""
    m = IFSC_PATTERN.search(text)
    if m:
        return m.group(1).upper()
    return ""


def _extract_micr(text: str) -> str:
    """Extract MICR code."""
    m = MICR_PATTERN.search(text)
    if m:
        return m.group(1)
    m2 = re.search(r"micr.*?(\d{9})\b", text, re.IGNORECASE)
    if m2:
        return m2.group(1)
    return ""


def _extract_bank_name(text: str, ifsc: str, bank_profile: Optional[dict]) -> str:
    """Extract bank name, with header text priority, IFSC prefix fallback, and profile fallback."""
    # 1. Look for Bank name in header text (e.g. "Sample National Bank", "State Bank of India")
    bm = re.search(r"\b([A-Za-z\s]{3,35}\s+Bank)\b", text[:400], re.IGNORECASE)
    if bm:
        b_name = bm.group(1).strip().title()
        if len(b_name) > 5 and not any(w in b_name.lower() for w in ["statement", "account", "opening"]):
            return b_name

    # 2. Derive from IFSC prefix
    if ifsc and len(ifsc) >= 4:
        prefix = ifsc[:4].upper()
        if prefix in _IFSC_BANK_MAP:
            return _IFSC_BANK_MAP[prefix]

    # 3. Profile fallback
    if bank_profile and bank_profile.get("bank_name") and bank_profile.get("bank_name").lower() not in ("generic", "none"):
        return str(bank_profile["bank_name"])

    # 4. Regex from text
    m = BANK_NAME_PATTERN.search(text)
    if m:
        name = m.group(1).strip().rstrip(",. ")
        if 2 < len(name) < 60:
            return name.title()

    return ""


def _extract_branch(text: str) -> str:
    """Extract branch name."""
    m = BRANCH_PATTERN.search(text)
    if m:
        branch = m.group(1).strip().rstrip(",. ")
        if 2 < len(branch) < 80:
            return branch.title()
    rev_m = re.search(r"([A-Za-z0-9\s]{3,30})\s+(?:branch(?:\s+name)?)\b", text, re.IGNORECASE)
    if rev_m:
        branch = rev_m.group(1).strip().rstrip(",. ")
        branch = re.sub(r"^\d+\s*", "", branch).strip()
        if 2 < len(branch) < 80:
            return branch.title()
    return ""


def _extract_customer_id(text: str) -> str:
    """Extract customer / CIF ID."""
    m = CUSTOMER_ID_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    return ""


def _normalise_account_type(raw: str) -> str:
    """Map raw account type text to a canonical label."""
    low = raw.lower()
    for keyword, canonical in _ACCOUNT_TYPE_KEYWORDS.items():
        if keyword in low:
            return canonical
    return raw.strip().title()


def _extract_account_type(text: str, bank_profile: Optional[dict]) -> str:
    """Extract and normalise account type."""
    if bank_profile and bank_profile.get("account_type"):
        return _normalise_account_type(str(bank_profile["account_type"]))

    m = ACCOUNT_TYPE_PATTERN.search(text)
    if m:
        return _normalise_account_type(m.group(1))
    return ""


def _extract_period(text: str) -> tuple[Optional[date], Optional[date]]:
    """Extract the statement period (from, to) dates."""
    # Try "Statement Period: DD/MM/YYYY to DD/MM/YYYY"
    m = STATEMENT_PERIOD_PATTERN.search(text)
    if m:
        date_from = normalize_date(m.group(1).strip())
        date_to = normalize_date(m.group(2).strip())
        if date_from or date_to:
            return date_from, date_to

    # Try "From: ... To: ..."
    m = PERIOD_FROM_TO_PATTERN.search(text)
    if m:
        date_from = normalize_date(m.group(1).strip())
        date_to = normalize_date(m.group(2).strip()) if m.group(2) else None
        if date_from or date_to:
            return date_from, date_to

    return None, None


def _extract_balance(text: str, pattern: re.Pattern[str]) -> Optional[Decimal]:
    """Generic balance extractor using a compiled pattern."""
    m = pattern.search(text)
    if m:
        raw = m.group(1).strip()
        return normalize_amount(raw)
    return None


def _extract_currency(text: str) -> str:
    """Extract ISO-4217 currency code from text, defaulting to INR."""
    m = CURRENCY_PATTERN.search(text)
    if m:
        return m.group(1).upper()
    return "INR"


def _extract_address(text: str, name: str) -> Optional[str]:
    """Attempt to locate an address block near the account holder name or via address anchor."""
    m_addr = ADDRESS_PATTERN.search(text)
    if m_addr:
        addr = m_addr.group(1).strip().rstrip(",. ")
        if len(addr) >= 10:
            return addr

    if not name:
        return None

    # Find position of name in text
    idx = text.lower().find(name.lower())
    if idx == -1:
        return None

    # Grab the next ~300 characters and look for a 6-digit PIN code
    snippet = text[idx + len(name): idx + len(name) + 300]
    if re.search(r"\b\d{6}\b", snippet):
        lines = [ln.strip() for ln in snippet.split("\n") if ln.strip()]
        address_lines = lines[:4]  # At most 4 lines
        address = ", ".join(address_lines)
        # Heuristic: must be at least 15 chars to be meaningful
        return address if len(address) >= 15 else None
    return None


def _extract_statement_date(text: str) -> Optional[date]:
    """Extract the statement generation/as-of date."""
    m = STATEMENT_DATE_PATTERN.search(text)
    if m:
        return normalize_date(m.group(1).strip())
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_account_details(
    text: str,
    bank_profile: Optional[dict] = None,
) -> AccountDetails:
    """Extract structured account metadata from raw bank statement text.

    Uses a combination of compiled regex patterns, keyword anchors, and
    positional heuristics. No machine-learning or external APIs are involved.

    Parameters
    ----------
    text:
        Full plain-text content of the bank statement (all pages
        concatenated, or just the header pages).
    bank_profile:
        Optional bank profile dict (from ``config/bank_profiles/``) that
        may supply ``bank_name``, ``account_type``, or layout hints to
        improve extraction accuracy.

    Returns
    -------
    :class:`AccountDetails`
        Populated dataclass. Fields that could not be extracted are left
        at their defaults (empty string or ``None``).
    """
    if not text:
        logger.warning("extract_account_details received empty text")
        return AccountDetails()

    details = AccountDetails()

    # --- Core identifiers ---
    details.account_number = _extract_account_number(text)
    details.ifsc = _extract_ifsc(text)
    details.micr = _extract_micr(text)
    details.customer_id = _extract_customer_id(text)

    # --- Institution ---
    details.bank_name = _extract_bank_name(text, details.ifsc, bank_profile)
    details.branch = _extract_branch(text)
    details.account_type = _extract_account_type(text, bank_profile)

    # --- Name ---
    details.account_holder_name = _extract_name(text, details.bank_name)

    # --- Address (best-effort) ---
    details.address = _extract_address(text, details.account_holder_name)

    # --- Period ---
    details.period_from, details.period_to = _extract_period(text)

    # --- Balances ---
    details.opening_balance = _extract_balance(text, OPENING_BALANCE_PATTERN)
    details.closing_balance = _extract_balance(text, CLOSING_BALANCE_PATTERN)

    # --- Currency ---
    details.currency = _extract_currency(text)

    # --- Statement date ---
    details.statement_date = _extract_statement_date(text)

    logger.debug("Extracted account details: %s", details.mask_for_log())

    if not details.is_complete():
        logger.warning(
            "Account details extraction incomplete — account_number='%s', bank='%s'",
            "XXXX" + details.account_number[-4:] if len(details.account_number) >= 4 else "XXXX",
            details.bank_name or "<unknown>",
        )

    return details
