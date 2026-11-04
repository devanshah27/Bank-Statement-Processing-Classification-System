"""Recurring transaction and inter-account transfer detection module."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
import statistics
from typing import Optional, Any

from .features import extract_features

logger = logging.getLogger(__name__)


@dataclass
class RecurringGroup:
    """Group of transactions recognized as recurring on a schedule."""
    merchant: str
    category: str
    transaction_indices: list[int]
    frequency: str  # "monthly", "weekly", "biweekly", "quarterly", "daily", "irregular"
    avg_amount: float
    amount_variance: float  # coefficient of variation
    last_date: Optional[str]
    next_expected_date: Optional[str]
    confidence: float


def detect_recurring(
    transactions: list[dict[str, Any]],
    min_occurrences: int = 2,
    amount_tolerance_pct: float = 0.20,  # amounts within 20%
) -> list[RecurringGroup]:
    """Detect repeating subscription, bill, rent, or salary payments.

    Args:
        transactions: List of transaction dicts with 'date', 'description', 'amount' (or debit/credit), 'category'.
        min_occurrences: Minimum number of occurrences to form a recurring pattern.
        amount_tolerance_pct: Relative difference allowed in recurring amounts.

    Returns:
        List of RecurringGroup objects sorted by confidence and frequency.
    """
    if len(transactions) < min_occurrences:
        return []

    # 1. Group transactions by normalized counterparty / pattern
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for idx, tx in enumerate(transactions):
        desc = str(tx.get("description", ""))
        debit = tx.get("debit")
        credit = tx.get("credit")
        amount = tx.get("amount")
        if amount is None:
            if debit not in (None, "", "0", "0.00"):
                amount = debit
                direction = "DR"
            else:
                amount = credit
                direction = "CR"
        else:
            direction = tx.get("direction", "DR")

        try:
            dec_amt = Decimal(str(amount)) if amount is not None else Decimal("0")
        except Exception:
            dec_amt = Decimal("0")

        feat = extract_features(desc, direction, dec_amt)
        key = feat.counterparty.strip().lower()
        if not key or len(key) < 3:
            # Fallback to cleaned description tokens
            key = feat.description_cleaned[:25].strip().lower()

        if key:
            groups[key].append((idx, tx))

    recurring_results: list[RecurringGroup] = []

    # 2. Analyze each group for regularity in interval and amount
    for merchant_key, items in groups.items():
        if len(items) < min_occurrences:
            continue

        indices = [it[0] for it in items]
        amounts: list[float] = []
        dates: list[date] = []
        categories: list[str] = []

        for _, tx in items:
            cat = str(tx.get("category", "Other/Uncategorized"))
            if cat != "Other/Uncategorized":
                categories.append(cat)

            amt_val = tx.get("amount", tx.get("debit", tx.get("credit", 0)))
            try:
                amounts.append(float(amt_val))
            except Exception:
                amounts.append(0.0)

            d_str = str(tx.get("date", "")).strip()
            if d_str:
                try:
                    d_obj = datetime.strptime(d_str[:10], "%Y-%m-%d").date()
                    dates.append(d_obj)
                except Exception:
                    pass

        if len(dates) < min_occurrences:
            continue

        # Sort by date
        sorted_pairs = sorted(zip(dates, amounts), key=lambda x: x[0])
        dates = [p[0] for p in sorted_pairs]
        amounts = [p[1] for p in sorted_pairs]

        avg_amount = statistics.mean(amounts) if amounts else 0.0
        stdev_amount = statistics.stdev(amounts) if len(amounts) > 1 else 0.0
        cv_amount = (stdev_amount / avg_amount) if avg_amount > 0 else 1.0

        # Intervals between consecutive dates
        intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        if not intervals:
            continue

        mean_interval = statistics.mean(intervals)
        stdev_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0.0

        # Frequency detection
        frequency = "irregular"
        next_date: Optional[date] = None
        confidence = 0.5

        if 25 <= mean_interval <= 35 and stdev_interval <= 7:
            frequency = "monthly"
            confidence = 0.90
            next_date = dates[-1] + timedelta(days=30)
        elif 6 <= mean_interval <= 8 and stdev_interval <= 2:
            frequency = "weekly"
            confidence = 0.90
            next_date = dates[-1] + timedelta(days=7)
        elif 13 <= mean_interval <= 16 and stdev_interval <= 3:
            frequency = "biweekly"
            confidence = 0.85
            next_date = dates[-1] + timedelta(days=14)
        elif 80 <= mean_interval <= 100 and stdev_interval <= 15:
            frequency = "quarterly"
            confidence = 0.80
            next_date = dates[-1] + timedelta(days=90)
        elif cv_amount < amount_tolerance_pct and len(amounts) >= 3:
            frequency = "monthly" if 20 <= mean_interval <= 40 else "irregular"
            confidence = 0.70
            next_date = dates[-1] + timedelta(days=int(mean_interval))

        if confidence >= 0.65:
            common_cat = max(set(categories), key=categories.count) if categories else "Utilities"
            recurring_results.append(
                RecurringGroup(
                    merchant=merchant_key.title(),
                    category=common_cat,
                    transaction_indices=indices,
                    frequency=frequency,
                    avg_amount=round(avg_amount, 2),
                    amount_variance=round(cv_amount, 3),
                    last_date=dates[-1].isoformat() if dates else None,
                    next_expected_date=next_date.isoformat() if next_date else None,
                    confidence=confidence,
                )
            )

    recurring_results.sort(key=lambda r: (-r.confidence, -len(r.transaction_indices)))
    return recurring_results

