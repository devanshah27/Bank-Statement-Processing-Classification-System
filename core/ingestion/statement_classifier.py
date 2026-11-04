"""Document type classifier module.

Distinguishes bank account statements from non-statement documents such as
invoices, purchase orders, credit card statements, and arbitrary documents.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentClassification:
    """Document classification result."""
    is_bank_statement: bool
    doc_type: str  # "bank_statement" | "credit_card_statement" | "invoice" | "empty" | "unknown"
    confidence: float
    reasons: list[str] = field(default_factory=list)


# Keywords indicative of Bank Account Statements
_BANK_STATEMENT_SIGNALS = [
    (r"\b(account\s+(?:no|number|num)|a/c\s+no)\b", 2.0),
    (r"\b(ifsc|micr|branch\s+code)\b", 2.0),
    (r"\b(opening\s+bal(?:ance)?|closing\s+bal(?:ance)?)\b", 2.5),
    (r"\b(statement\s+of\s+account|account\s+statement|bank\s+statement|passbook)\b", 2.5),
    (r"\b(savings\s+account|current\s+account|sb\s+a/c|ca\s+a/c)\b", 2.0),
    (r"\b(withdrawal|deposit|debit|credit|running\s+balance|dr/cr)\b", 1.5),
    (r"\b(value\s+date|txn\s+date|transaction\s+date|narration|chq/ref)\b", 1.5),
]

# Keywords indicative of Credit Card Statements
_CREDIT_CARD_SIGNALS = [
    (r"\b(credit\s+card\s+statement|card\s+statement|credit\s+card\s+account)\b", 3.0),
    (r"\b(payment\s+due\s+date|min(?:imum)?\s+amount\s+due|min\s+due)\b", 3.0),
    (r"\b(credit\s+limit|available\s+credit|available\s+cash\s+limit)\b", 2.5),
    (r"\b(reward\s+points|points\s+earned|points\s+redeemed)\b", 2.0),
    (r"\b(card\s+no|card\s+number|primary\s+card|cardmember)\b", 2.0),
]

# Keywords indicative of Invoices / Bills
_INVOICE_SIGNALS = [
    (r"\b(tax\s+invoice|commercial\s+invoice|proforma\s+invoice)\b", 3.5),
    (r"\b(invoice\s+no|invoice\s+number|inv\s+no|bill\s+no)\b", 2.5),
    (r"\b(bill\s+to|ship\s+to|consignee|vendor|supplier)\b", 2.5),
    (r"\b(purchase\s+order|p\.?o\.?\s+no|po\s+number)\b", 2.0),
    (r"\b(subtotal|gstin|cgst|sgst|igst|hsn/sac|hsn\s+code|taxable\s+amount)\b", 2.0),
    (r"\b(terms\s+of\s+delivery|dispatch\s+through|destination)\b", 1.5),
]


def classify_document(text_sample: str, page_count: int = 1) -> DocumentClassification:
    """Classify document text to determine if it is a genuine bank account statement.

    Args:
        text_sample: Extracted text from initial pages of the document.
        page_count: Total number of pages in the document.

    Returns:
        DocumentClassification detailing detected doc_type and whether it should be parsed.
    """
    clean_text = text_sample.lower()
    if not clean_text or clean_text.isspace():
        return DocumentClassification(
            is_bank_statement=False,
            doc_type="empty",
            confidence=1.0,
            reasons=["Document contains no readable text or words."]
        )

    bank_score = 0.0
    cc_score = 0.0
    invoice_score = 0.0

    matched_bank: list[str] = []
    matched_cc: list[str] = []
    matched_inv: list[str] = []

    for pattern, weight in _BANK_STATEMENT_SIGNALS:
        if re.search(pattern, clean_text, re.IGNORECASE):
            bank_score += weight
            matched_bank.append(pattern)

    for pattern, weight in _CREDIT_CARD_SIGNALS:
        if re.search(pattern, clean_text, re.IGNORECASE):
            cc_score += weight
            matched_cc.append(pattern)

    for pattern, weight in _INVOICE_SIGNALS:
        if re.search(pattern, clean_text, re.IGNORECASE):
            invoice_score += weight
            matched_inv.append(pattern)

    logger.debug(
        "Document classification scores - Bank: %.1f, CC: %.1f, Invoice: %.1f",
        bank_score, cc_score, invoice_score
    )

    # 1. Invoice detection: high invoice score, distinctly higher than bank score
    if invoice_score >= 4.0 and invoice_score > bank_score:
        return DocumentClassification(
            is_bank_statement=False,
            doc_type="invoice",
            confidence=min(0.99, invoice_score / 6.0),
            reasons=[
                "Document detected as an invoice/bill.",
                f"Matched invoice features: {len(matched_inv)} patterns."
            ]
        )

    # 2. Credit Card Statement: presence of distinct credit card keywords
    if cc_score >= 4.0 and cc_score > bank_score:
        return DocumentClassification(
            is_bank_statement=False,
            doc_type="credit_card_statement",
            confidence=min(0.99, cc_score / 6.0),
            reasons=[
                "Document detected as a credit card statement (no running balance ledger).",
                f"Matched credit card features: {len(matched_cc)} patterns."
            ]
        )

    # 3. Bank Account Statement: strong indicators
    if bank_score >= 3.5:
        return DocumentClassification(
            is_bank_statement=True,
            doc_type="bank_statement",
            confidence=min(0.99, bank_score / 7.0),
            reasons=[f"Matched bank statement indicators: {len(matched_bank)} patterns."]
        )

    # 4. Borderline / Unknown
    if bank_score > 1.5:
        return DocumentClassification(
            is_bank_statement=True,
            doc_type="bank_statement",
            confidence=0.60,
            reasons=["Weak bank statement signals found; proceeding with heuristic parsing."]
        )

    return DocumentClassification(
        is_bank_statement=False,
        doc_type="unknown",
        confidence=0.40,
        reasons=["Document does not exhibit standard bank statement patterns."]
    )

