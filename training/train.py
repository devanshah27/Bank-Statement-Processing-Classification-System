"""Train the transaction classification model and export to numpy format."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from core.classification.features import extract_features
from training.export_model import export_model_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and prepare training data."""
    logger.info("Loading training data from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Feature engineering for each row
    cleaned_descs = []
    directions = []
    channels = []
    log_amounts = []

    for _, row in df.iterrows():
        desc = str(row.get("description", ""))
        direction = str(row.get("direction", "DR")).upper()
        amt_raw = row.get("amount", 0.0)
        try:
            amt = float(amt_raw)
        except Exception:
            amt = 0.0

        from decimal import Decimal
        dec_amt = Decimal(str(amt)) if amt is not None else None
        feat = extract_features(desc, direction, dec_amt)
        cleaned_descs.append(f"{feat.description_cleaned} {feat.counterparty}".strip())
        directions.append(direction if direction in ("DR", "CR") else "DR")
        channels.append(feat.payment_channel)
        log_amounts.append(math.log1p(max(0.0, amt)))

    df["feat_text"] = cleaned_descs
    df["feat_direction"] = directions
    df["feat_channel"] = channels
    df["feat_log_amount"] = log_amounts

    return df


def build_pipeline(classifier_type: str = "lr") -> Pipeline:
    """Construct full preprocessing + classification pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=15000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                ),
                "feat_text",
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    ngram_range=(3, 4),
                    analyzer="char_wb",
                    max_features=10000,
                    sublinear_tf=True,
                ),
                "feat_text",
            ),
            (
                "direction_ohe",
                OneHotEncoder(categories=[["DR", "CR"]], handle_unknown="ignore"),
                ["feat_direction"],
            ),
            (
                "channel_ohe",
                OneHotEncoder(handle_unknown="ignore"),
                ["feat_channel"],
            ),
        ],
        remainder="passthrough",  # passes log_amount
    )

    if classifier_type == "lr":
        base_clf = LogisticRegression(C=2.0, max_iter=1000, random_state=SEED)
    elif classifier_type == "svm":
        base_clf = LinearSVC(C=1.0, max_iter=2000, random_state=SEED)
    elif classifier_type == "nb":
        base_clf = ComplementNB(alpha=0.5)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", calibrated_clf),
    ])


def main():
    parser = argparse.ArgumentParser(description="Train bank statement transaction classifier.")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("samples/test_dataset/classification_data/classification_train_labeled.csv"),
    )
    parser.add_argument(
        "--holdout-file",
        type=Path,
        default=Path("samples/test_dataset/classification_data/classification_holdout_test_noisy.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/v1"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    df_train = load_dataset(args.train_file)
    X_train = df_train[["feat_text", "feat_direction", "feat_channel", "feat_log_amount"]]
    y_train = df_train["category"]
    classes = sorted(list(y_train.unique()))

    logger.info("Dataset shape: %s with %d distinct classes", X_train.shape, len(classes))

    # Cross-validation comparison between LR, SVM, and ComplementNB
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model_scores = {}

    for m_type in ["lr", "svm", "nb"]:
        logger.info("Evaluating %s with 5-fold cross-validation...", m_type.upper())
        pipe = build_pipeline(m_type)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        model_scores[m_type] = (mean_score, std_score)
        logger.info("Model %s: Macro F1 = %.4f (+/- %.4f)", m_type.upper(), mean_score, std_score)

    best_model_type = max(model_scores, key=lambda k: model_scores[k][0])
    logger.info("Best model selected: %s (F1: %.4f)", best_model_type.upper(), model_scores[best_model_type][0])

    # Fit best model on complete training dataset
    final_pipeline = build_pipeline(best_model_type)
    final_pipeline.fit(X_train, y_train)

    train_preds = final_pipeline.predict(X_train)
    train_macro_f1 = float(f1_score(y_train, train_preds, average="macro"))
    logger.info("Train set Macro F1: %.4f", train_macro_f1)

    # Export to numpy-only format
    metadata = {
        "version": "v1.0.0",
        "created_at": datetime.datetime.now().isoformat(),
        "model_type": best_model_type,
        "n_samples": len(df_train),
        "n_classes": len(classes),
        "classes": classes,
        "cv_scores": {k: f"{v[0]:.4f} +/- {v[1]:.4f}" for k, v in model_scores.items()},
        "train_macro_f1": train_macro_f1,
    }

    export_model_artifacts(final_pipeline, classes, args.output_dir, metadata)

    # Save evaluation report
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = args.reports_dir / f"model_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    cls_report = classification_report(y_train, train_preds, digits=4)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Transaction Classification Model Training Report\n\n")
        f.write(f"- **Trained At**: {metadata['created_at']}\n")
        f.write(f"- **Training Samples**: {len(df_train)}\n")
        f.write(f"- **Classes**: {len(classes)}\n")
        f.write(f"- **Selected Architecture**: {best_model_type.upper()} with TF-IDF (word+char) + OneHot Features\n\n")
        f.write("## 5-Fold Cross-Validation Model Comparison\n\n")
        f.write("| Model | Macro F1 | Std Dev |\n|---|---|---|\n")
        for k, v in model_scores.items():
            f.write(f"| {k.upper()} | {v[0]:.4f} | {v[1]:.4f} |\n")
        f.write(f"\n## Training Classification Report\n\n```\n{cls_report}\n```\n")

    logger.info("Saved training report to %s", report_file)


if __name__ == "__main__":
    main()
