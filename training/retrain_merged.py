"""Retrain model with all available datasets merged and export artifacts."""

from __future__ import annotations

import glob
import logging
import math
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from decimal import Decimal

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from core.classification.features import extract_features
from training.export_model import export_model_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


def load_all_merged_data() -> pd.DataFrame:
    """Load and merge all training and ground truth datasets."""
    dfs = []
    
    # 1. Primary classification training data
    train_csv = PROJECT_ROOT / "bank_statement_test_dataset" / "classification_data" / "classification_train_labeled.csv"
    if train_csv.exists():
        logger.info("Loading %s", train_csv)
        dfs.append(pd.read_csv(train_csv))

    # 2. Holdout noisy data
    holdout_csv = PROJECT_ROOT / "bank_statement_test_dataset" / "classification_data" / "classification_holdout_test_noisy.csv"
    if holdout_csv.exists():
        logger.info("Loading %s", holdout_csv)
        dfs.append(pd.read_csv(holdout_csv))

    # 3. Ground truth statement CSVs
    gt_dir = PROJECT_ROOT / "bank_statement_test_dataset" / "ground_truth"
    gt_rows = []
    for f in gt_dir.glob("*.csv"):
        try:
            d = pd.read_csv(f)
            if "description" in d.columns and "category" in d.columns:
                for _, r in d.iterrows():
                    cat = str(r["category"]).strip()
                    desc = str(r["description"]).strip()
                    if cat and cat not in ("nan", "None", ""):
                        deb = r.get("debit")
                        cred = r.get("credit")
                        amt = 0.0
                        direction = "DR"
                        if pd.notna(deb):
                            try:
                                d_val = float(str(deb).replace(",", "").strip())
                                if d_val > 0:
                                    amt = d_val
                                    direction = "DR"
                            except Exception:
                                pass
                        if amt == 0.0 and pd.notna(cred):
                            try:
                                c_val = float(str(cred).replace(",", "").strip())
                                if c_val > 0:
                                    amt = c_val
                                    direction = "CR"
                            except Exception:
                                pass
                        gt_rows.append({
                            "description": desc,
                            "direction": direction,
                            "amount": amt,
                            "category": cat
                        })
        except Exception as e:
            logger.warning("Error reading %s: %s", f, e)

    if gt_rows:
        logger.info("Loaded %d rows from ground truth CSVs", len(gt_rows))
        dfs.append(pd.DataFrame(gt_rows))

    df_merged = pd.concat(dfs, ignore_index=True)
    df_merged = df_merged.dropna(subset=["description", "category"])
    df_merged = df_merged.drop_duplicates(subset=["description", "direction", "amount", "category"])
    logger.info("Total merged unique training samples: %d", len(df_merged))

    # Feature extraction
    cleaned_descs = []
    directions = []
    channels = []
    log_amounts = []

    for _, row in df_merged.iterrows():
        desc = str(row.get("description", ""))
        direction = str(row.get("direction", "DR")).upper().strip()
        amt_raw = row.get("amount", 0.0)
        try:
            amt = float(amt_raw)
        except Exception:
            amt = 0.0

        dec_amt = Decimal(str(amt)) if amt is not None else None
        feat = extract_features(desc, direction, dec_amt)
        cleaned_descs.append(f"{feat.description_cleaned} {feat.counterparty}".strip())
        directions.append(direction if direction in ("DR", "CR") else "DR")
        channels.append(feat.payment_channel)
        log_amounts.append(math.log1p(max(0.0, amt)))

    df_merged["feat_text"] = cleaned_descs
    df_merged["feat_direction"] = directions
    df_merged["feat_channel"] = channels
    df_merged["feat_log_amount"] = log_amounts

    return df_merged


def build_pipeline() -> Pipeline:
    """Construct full preprocessing + calibrated classification pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=25000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                ),
                "feat_text",
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    ngram_range=(3, 5),
                    analyzer="char_wb",
                    max_features=18000,
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
        remainder="passthrough",
    )

    base_clf = LogisticRegression(C=3.0, max_iter=1500, random_state=SEED)
    calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", calibrated_clf),
    ])


def main():
    df = load_all_merged_data()
    X = df[["feat_text", "feat_direction", "feat_channel", "feat_log_amount"]]
    y = df["category"]
    classes = sorted(list(y.unique()))

    logger.info("Dataset shape: %s with %d distinct classes: %s", X.shape, len(classes), classes)

    pipeline = build_pipeline()
    logger.info("Fitting calibrated model on full merged dataset (%d rows)...", len(df))
    pipeline.fit(X, y)

    train_preds = pipeline.predict(X)
    train_macro_f1 = float(f1_score(y, train_preds, average="macro"))
    logger.info("Training Macro F1: %.4f", train_macro_f1)
    print(classification_report(y, train_preds, digits=4))

    # Export to models/v1
    output_dir = PROJECT_ROOT / "models" / "v1"
    metadata = {
        "version": "v1.1.0-merged-dataset",
        "created_at": pd.Timestamp.now().isoformat(),
        "model_type": "lr",
        "n_samples": len(df),
        "n_classes": len(classes),
        "classes": classes,
        "train_macro_f1": train_macro_f1,
    }

    export_model_artifacts(pipeline, classes, output_dir, metadata)
    logger.info("Successfully exported retrained model artifacts to %s", output_dir)


if __name__ == "__main__":
    main()

