"""Evaluate the hybrid classification pipeline on the noisy holdout dataset."""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decimal import Decimal
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

from core.classification.hybrid_classifier import HybridClassifier
from core.classification.rules_engine import RulesEngine
from core.classification.ml_model import NumpyClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_classifier(
    holdout_csv: Path,
    models_dir: Path,
    reports_dir: Path,
) -> dict[str, float]:
    """Run evaluation on noisy holdout dataset and write markdown report."""
    logger.info("Evaluating classifier on %s", holdout_csv)
    df = pd.read_csv(holdout_csv)

    rules_engine = RulesEngine()
    ml_model = NumpyClassifier(models_dir)
    classifier = HybridClassifier(rules_engine=rules_engine, ml_model=ml_model)

    y_true = []
    y_pred = []
    sources = []
    confidences = []

    for _, row in df.iterrows():
        desc = str(row.get("description", ""))
        direction = str(row.get("direction", "DR")).upper()
        amt_raw = row.get("amount", 0.0)
        try:
            amt = Decimal(str(amt_raw))
        except Exception:
            amt = Decimal("0")

        true_cat = str(row.get("category", "")).strip()
        result = classifier.classify(desc, direction, amt)

        y_true.append(true_cat)
        y_pred.append(result.category)
        sources.append(result.source)
        confidences.append(result.confidence)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    rule_count = sum(1 for s in sources if "rule" in s)
    ml_count = sum(1 for s in sources if s == "ml")
    uncat_count = sum(1 for s in sources if s == "uncategorized")
    total = len(df)

    logger.info("Holdout Evaluation: Accuracy = %.4f, Macro F1 = %.4f", acc, macro_f1)
    logger.info(
        "Source coverage: Rules: %d (%.1f%%), ML: %d (%.1f%%), Uncategorized: %d (%.1f%%)",
        rule_count, (rule_count / total) * 100,
        ml_count, (ml_count / total) * 100,
        uncat_count, (uncat_count / total) * 100,
    )

    cls_rep = classification_report(y_true, y_pred, digits=4, zero_division=0)

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "model_evaluation_holdout.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Model Evaluation on Noisy Holdout Dataset\n\n")
        f.write(f"- **Evaluated At**: {datetime.datetime.now().isoformat()}\n")
        f.write(f"- **Holdout File**: `{holdout_csv.name}`\n")
        f.write(f"- **Test Rows**: {total}\n")
        f.write(f"- **Accuracy**: **{acc:.2%}**\n")
        f.write(f"- **Macro F1**: **{macro_f1:.4f}**\n")
        f.write(f"- **Weighted F1**: **{weighted_f1:.4f}**\n\n")
        f.write("## Hybrid Engine Coverage Breakdown\n\n")
        f.write(f"- **Stage A (Rules / Heuristics)**: {rule_count} rows ({rule_count/total:.1%})\n")
        f.write(f"- **Stage B (ML Model)**: {ml_count} rows ({ml_count/total:.1%})\n")
        f.write(f"- **Uncategorized / Needs Review**: {uncat_count} rows ({uncat_count/total:.1%})\n\n")
        f.write("## Per-Class Classification Report\n\n")
        f.write("```\n")
        f.write(cls_rep)
        f.write("\n```\n")

    logger.info("Saved evaluation report to %s", report_path)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "rule_coverage": rule_count / total,
        "ml_coverage": ml_count / total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier on holdout dataset.")
    parser.add_argument(
        "--holdout-file",
        type=Path,
        default=Path("samples/test_dataset/classification_data/classification_holdout_test_noisy.csv"),
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models/v1"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    evaluate_classifier(args.holdout_file, args.models_dir, args.reports_dir)


if __name__ == "__main__":
    main()
