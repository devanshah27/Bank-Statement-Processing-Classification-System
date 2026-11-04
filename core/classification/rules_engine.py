"""YAML-driven priority-ordered rule engine for transaction classification."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Optional, Any
import yaml
from rapidfuzz import fuzz

from .features import TransactionFeatures

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


@dataclass
class Rule:
    """Single classification rule."""
    id: str
    priority: int
    name: str
    conditions: list[dict[str, Any]]
    category: str
    confidence: float
    reason: str
    enabled: bool = True


@dataclass
class RuleResult:
    """Result of rule-based evaluation."""
    matched: bool
    category: str
    confidence: float
    rule_id: str
    rule_name: str
    reason: str


class RulesEngine:
    """Config-driven rule engine supporting regex, keyword, and fuzzy merchant matching."""

    def __init__(
        self,
        rules_path: Optional[Path] = None,
        merchants_path: Optional[Path] = None,
    ):
        self.rules_path = rules_path or (DEFAULT_CONFIG_DIR / "rules.yaml")
        self.merchants_path = merchants_path or (DEFAULT_CONFIG_DIR / "merchants.yaml")
        self.rules: list[Rule] = []
        self.merchants: list[dict[str, Any]] = []
        self._compiled_patterns: dict[str, re.Pattern[str]] = {}
        self.reload()

    def reload(self) -> None:
        """Load or reload rules and merchant dictionaries from YAML."""
        self.rules.clear()
        self.merchants.clear()
        self._compiled_patterns.clear()

        # Load rules
        if self.rules_path.is_file():
            try:
                with open(self.rules_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                raw_rules = data.get("rules", [])
                for r in raw_rules:
                    rule = Rule(
                        id=str(r.get("id", "")),
                        priority=int(r.get("priority", 100)),
                        name=str(r.get("name", "")),
                        conditions=r.get("conditions", []),
                        category=str(r.get("category", "")),
                        confidence=float(r.get("confidence", 0.8)),
                        reason=str(r.get("reason", "")),
                        enabled=bool(r.get("enabled", True)),
                    )
                    self.rules.append(rule)
                self.rules.sort(key=lambda x: x.priority)
                logger.info("Loaded %d classification rules.", len(self.rules))
            except Exception as e:
                logger.error("Failed to load rules from %s: %s", self.rules_path, e)
        else:
            logger.warning("Rules file not found at %s", self.rules_path)

        # Load merchants
        if self.merchants_path.is_file():
            try:
                with open(self.merchants_path, "r", encoding="utf-8") as f:
                    m_data = yaml.safe_load(f) or {}
                self.merchants = m_data.get("merchants", [])
                logger.info("Loaded %d merchant profiles.", len(self.merchants))
            except Exception as e:
                logger.error("Failed to load merchants from %s: %s", self.merchants_path, e)

    def _get_pattern(self, pattern_str: str) -> re.Pattern[str]:
        if pattern_str not in self._compiled_patterns:
            self._compiled_patterns[pattern_str] = re.compile(pattern_str, re.IGNORECASE)
        return self._compiled_patterns[pattern_str]

    def _evaluate_condition(self, cond: dict[str, Any], features: TransactionFeatures) -> bool:
        c_type = cond.get("type", "")
        c_val = cond.get("value")

        if c_type == "direction":
            return features.direction.upper() == str(c_val).upper()

        if c_type == "payment_mode":
            return features.payment_channel.upper() == str(c_val).upper()

        if c_type == "pattern":
            pat = self._get_pattern(str(c_val))
            # Test against cleaned, normalized, and raw counterparty
            target_str = f"{features.description_normalized} {features.counterparty}"
            return bool(pat.search(target_str))

        if c_type == "description_contains":
            target_str = features.description_normalized.lower()
            if isinstance(c_val, list):
                return any(str(kw).lower() in target_str for kw in c_val)
            return str(c_val).lower() in target_str

        if c_type == "counterparty_contains":
            target_str = features.counterparty.lower()
            if isinstance(c_val, list):
                return any(str(kw).lower() in target_str for kw in c_val)
            return str(c_val).lower() in target_str

        if c_type == "amount_min":
            try:
                return features.amount >= Decimal(str(c_val))
            except Exception:
                return False

        if c_type == "amount_max":
            try:
                return features.amount <= Decimal(str(c_val))
            except Exception:
                return False

        return False

    def _check_merchants(self, features: TransactionFeatures) -> Optional[RuleResult]:
        """Check merchant dictionary using exact substring and rapidfuzz."""
        desc_lower = features.description_normalized.lower()
        counterparty_lower = features.counterparty.lower()
        search_corpus = f"{desc_lower} {counterparty_lower}"
        corpus_no_vpa = re.sub(r"@[a-z0-9]+", " ", search_corpus)

        best_match: Optional[dict[str, Any]] = None
        best_score = 0.0

        for m in self.merchants:
            name = m.get("name", "")
            aliases = m.get("aliases", [name])
            category = m.get("category", "")
            corpus_to_use = (
                corpus_no_vpa
                if name.lower() in ("paytm", "phonepe", "google pay", "gpay")
                else search_corpus
            )

            # 1. Exact alias match in string
            for alias in aliases:
                alias_clean = alias.strip().lower()
                if not alias_clean:
                    continue
                # Substring check with word boundary so e.g. "cred" doesn't match inside "credit"
                alias_pat = self._get_pattern(r"\b" + re.escape(alias_clean) + r"\b")
                if alias_pat.search(corpus_to_use):
                    return RuleResult(
                        matched=True,
                        category=category,
                        confidence=0.96,
                        rule_id=f"merchant_{name.lower().replace(' ', '_')}",
                        rule_name=f"Merchant match: {name}",
                        reason=f"Matched merchant '{name}' alias '{alias_clean}'",
                    )

            # 2. Fuzzy match on counterparty words if counterparty is reasonably long
            counterparty_words = counterparty_lower.split()
            for alias in aliases:
                alias_clean = alias.strip().lower()
                if len(alias_clean) >= 4:
                    for w in counterparty_words:
                        if len(w) >= 4:
                            ratio = fuzz.ratio(alias_clean, w)
                            if ratio > best_score and ratio >= 88:
                                best_score = ratio
                                best_match = m

        if best_match and best_score >= 88:
            return RuleResult(
                matched=True,
                category=best_match.get("category", ""),
                confidence=min(0.92, 0.70 + (best_score / 100.0) * 0.25),
                rule_id=f"fuzzy_merchant_{best_match.get('name', '').lower()}",
                rule_name=f"Fuzzy Merchant: {best_match.get('name')}",
                reason=f"Fuzzy matched '{best_match.get('name')}' with score {best_score:.0f}",
            )

        return None

    def classify(self, features: TransactionFeatures) -> RuleResult:
        """Classify transaction features using priority rules and merchant mapping."""
        # 1. Evaluate top-priority rules (priority < 30: refund, bank fees, atm)
        for rule in self.rules:
            if not rule.enabled or rule.priority >= 30:
                continue

            all_matched = True
            for cond in rule.conditions:
                if not self._evaluate_condition(cond, features):
                    all_matched = False
                    break

            if all_matched:
                return RuleResult(
                    matched=True,
                    category=rule.category,
                    confidence=rule.confidence,
                    rule_id=rule.id,
                    rule_name=rule.name,
                    reason=rule.reason or f"Matched rule '{rule.name}'",
                )

        # 2. Check merchant dictionary for high-confidence specific matches
        merchant_res = self._check_merchants(features)
        if merchant_res and merchant_res.confidence >= 0.90:
            return merchant_res

        # 3. Evaluate remaining priority-ordered rules (priority >= 30)
        for rule in self.rules:
            if not rule.enabled or rule.priority < 30:
                continue

            all_matched = True
            for cond in rule.conditions:
                if not self._evaluate_condition(cond, features):
                    all_matched = False
                    break

            if all_matched:
                return RuleResult(
                    matched=True,
                    category=rule.category,
                    confidence=rule.confidence,
                    rule_id=rule.id,
                    rule_name=rule.name,
                    reason=rule.reason or f"Matched rule '{rule.name}'",
                )

        # Fallback to lower-confidence merchant match if any was found
        if merchant_res:
            return merchant_res

        return RuleResult(
            matched=False,
            category="Other/Uncategorized",
            confidence=0.0,
            rule_id="none",
            rule_name="No rule matched",
            reason="No heuristic rule or merchant alias matched",
        )
