"""Retrain classification model incorporating human-in-the-loop corrections."""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from training.train import load_dataset, build_pipeline, SEED
from training.export_model import export_model_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Retrain model with user feedback corrections.")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("samples/test_dataset/classification_data/classification_train_labeled.csv"),
    )
    parser.add_argument(
        "--feedback-file",
        type=Path,
        default=Path("data/user_feedback.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/v1"))
    args = parser.parse_args()

    df_train = pd.read_csv(args.train_file)
    logger.info("Base training data: %d rows", len(df_train))

    if args.feedback_file.is_file():
        df_fb = pd.read_csv(args.feedback_file)
        if not df_fb.empty:
            logger.info("Found %d user feedback corrections", len(df_fb))
            # Format feedback into same schema
            fb_rows = []
            for _, r in df_fb.iterrows():
                fb_rows.append({
                    "description": r.get("description", ""),
                    "direction": r.get("direction", "DR"),
                    "amount": r.get("amount", 0.0),
                    "category": r.get("new_category", r.get("category", "")),
                })
            df_fb_clean = pd.DataFrame(fb_rows)
            # Give user feedback 3x weight by duplicating
            df_combined = pd.concat([df_train, df_fb_clean, df_fb_clean, df_fb_clean], ignore_index=True)
            logger.info("Combined training data size: %d rows", len(df_combined))
        else:
            df_combined = df_train
    else:
        logger.info("No user feedback file found at %s. Retraining on base data.", args.feedback_file)
        df_combined = df_train

    # Save temporary merged dataset and retrain
    temp_merged = Path("data/merged_train_feedback.csv")
    temp_merged.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(temp_merged, index=False)

    df_prepared = load_dataset(temp_merged)
    X = df_prepared[["feat_text", "feat_direction", "feat_channel", "feat_log_amount"]]
    y = df_prepared["category"]
    classes = sorted(list(y.unique()))

    pipeline = build_pipeline("lr")
    pipeline.fit(X, y)

    meta = {
        "version": f"v1-feedback-{datetime.datetime.now().strftime('%Y%m%d%H%M')}",
        "created_at": datetime.datetime.now().isoformat(),
        "model_type": "lr_with_feedback",
        "n_samples": len(df_combined),
        "n_classes": len(classes),
        "classes": classes,
    }

    export_model_artifacts(pipeline, classes, args.output_dir, meta)
    logger.info("Feedback retraining complete. Model updated at %s", args.output_dir)


if __name__ == "__main__":
    main()
