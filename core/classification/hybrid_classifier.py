"""Hybrid classifier orchestrator combining Stage A (rules) and Stage B (ML)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Any

from .features import extract_features, TransactionFeatures
from .rules_engine import RulesEngine, RuleResult
from .ml_model import NumpyClassifier, MLPrediction

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Final unified classification result for a transaction."""
    category: str
    confidence: float
    source: str  # "rule" | "ml" | "uncategorized"
    rule_id: Optional[str] = None
    rule_name: Optional[str] = None
    ml_category: Optional[str] = None
    ml_confidence: Optional[float] = None
    reason: str = ""


class HybridClassifier:
    """Two-stage hybrid classifier: Heuristic Rule Engine + Numpy ML Inference."""

    def __init__(
        self,
        rules_engine: Optional[RulesEngine] = None,
        ml_model: Optional[NumpyClassifier] = None,
        rule_threshold: float = 0.75,
        ml_threshold: float = 0.15,
    ):
        self.rules_engine = rules_engine or RulesEngine()
        self.ml_model = ml_model or NumpyClassifier()
        self.rule_threshold = rule_threshold
        self.ml_threshold = ml_threshold

    def classify(
        self,
        description: str,
        direction: str = "DR",
        amount: Optional[Decimal | float] = None,
    ) -> ClassificationResult:
        """Classify a single transaction with Stage A heuristics, then Stage B ML.

        Args:
            description: Narration or transaction description text.
            direction: "DR" (Debit) or "CR" (Credit).
            amount: Numeric transaction amount.

        Returns:
            ClassificationResult with assigned category and origin metadata.
        """
        # Convert amount to Decimal
        dec_amount: Optional[Decimal] = None
        if amount is not None:
            try:
                dec_amount = Decimal(str(amount))
            except Exception:
                dec_amount = None

        features = extract_features(description, direction, dec_amount)

        # Stage A: Rule engine
        rule_res: RuleResult = self.rules_engine.classify(features)

        # If rule matched with sufficient confidence, rule wins
        if rule_res.matched and rule_res.confidence >= self.rule_threshold:
            return ClassificationResult(
                category=rule_res.category,
                confidence=rule_res.confidence,
                source="rule",
                rule_id=rule_res.rule_id,
                rule_name=rule_res.rule_name,
                reason=rule_res.reason,
            )

        # Stage B: ML model
        ml_res: MLPrediction = self.ml_model.predict(features)

        if self.ml_model.is_loaded and ml_res.confidence >= self.ml_threshold:
            # Check if rule had a partial/lower-confidence match that agrees
            reason = f"ML predicted {ml_res.category} (confidence: {ml_res.confidence:.1%})"
            if rule_res.matched and rule_res.category == ml_res.category:
                combined_conf = min(0.98, max(rule_res.confidence, ml_res.confidence) + 0.05)
                reason += f" and corroborated by rule '{rule_res.rule_name}'"
                return ClassificationResult(
                    category=ml_res.category,
                    confidence=combined_conf,
                    source="rule+ml",
                    rule_id=rule_res.rule_id,
                    rule_name=rule_res.rule_name,
                    ml_category=ml_res.category,
                    ml_confidence=ml_res.confidence,
                    reason=reason,
                )

            return ClassificationResult(
                category=ml_res.category,
                confidence=ml_res.confidence,
                source="ml",
                ml_category=ml_res.category,
                ml_confidence=ml_res.confidence,
                reason=reason,
            )

        # If rule had any match even below threshold, use it as fallback
        if rule_res.matched and rule_res.confidence > 0.0:
            return ClassificationResult(
                category=rule_res.category,
                confidence=rule_res.confidence,
                source="rule",
                rule_id=rule_res.rule_id,
                rule_name=rule_res.rule_name,
                ml_category=ml_res.category if self.ml_model.is_loaded else None,
                ml_confidence=ml_res.confidence if self.ml_model.is_loaded else None,
                reason=f"Low-confidence rule match: {rule_res.reason}",
            )

        # Below all thresholds
        return ClassificationResult(
            category="Other/Uncategorized",
            confidence=max(rule_res.confidence, ml_res.confidence if self.ml_model.is_loaded else 0.0),
            source="uncategorized",
            ml_category=ml_res.category if self.ml_model.is_loaded else None,
            ml_confidence=ml_res.confidence if self.ml_model.is_loaded else None,
            reason="Confidence below threshold; requires manual review",
        )

    def classify_batch(
        self,
        transactions: list[dict[str, Any]]
    ) -> list[ClassificationResult]:
        """Classify a list of transaction dictionaries in sequence."""
        results = []
        for t in transactions:
            desc = str(t.get("description", ""))
            # Determine direction: explicit or inferred from debit/credit
            direction = str(t.get("direction", "")).upper()
            if not direction:
                debit = t.get("debit")
                credit = t.get("credit")
                if debit is not None and str(debit).strip() not in ("", "0", "0.00"):
                    direction = "DR"
                    amount = debit
                elif credit is not None and str(credit).strip() not in ("", "0", "0.00"):
                    direction = "CR"
                    amount = credit
                else:
                    direction = "DR"
                    amount = t.get("amount", 0)
            else:
                amount = t.get("amount", t.get("debit", t.get("credit", 0)))

            res = self.classify(desc, direction, amount)
            results.append(res)
        return results
