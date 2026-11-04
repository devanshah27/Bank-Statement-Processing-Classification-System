"""Export trained scikit-learn models to portable numpy format for serving."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


def export_model_artifacts(
    pipeline: Any,
    classes: list[str],
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """Extract model parameters into numpy arrays and json vocabulary files.

    Args:
        pipeline: Trained sklearn Pipeline or CalibratedClassifierCV.
        classes: List of string class names.
        output_dir: Directory where artifacts will be saved.
        metadata: Metadata dictionary to save in metadata.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract vectorizers and preprocessors from ColumnTransformer
    col_trans = pipeline.named_steps["preprocessor"]
    word_vec = col_trans.named_transformers_["word_tfidf"]
    char_vec = col_trans.named_transformers_.get("char_tfidf")
    dir_enc = col_trans.named_transformers_.get("direction_ohe")
    chan_enc = col_trans.named_transformers_.get("channel_ohe")

    # Word TF-IDF
    vocab = {str(k): int(v) for k, v in word_vec.vocabulary_.items()}
    with open(output_dir / "tfidf_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    np.save(output_dir / "tfidf_idf.npy", word_vec.idf_.astype(np.float32))

    # Char TF-IDF
    if char_vec is not None:
        char_vocab = {str(k): int(v) for k, v in char_vec.vocabulary_.items()}
        with open(output_dir / "char_tfidf_vocab.json", "w", encoding="utf-8") as f:
            json.dump(char_vocab, f)
        np.save(output_dir / "char_tfidf_idf.npy", char_vec.idf_.astype(np.float32))

    # OneHot encoders categories
    direction_cats = list(dir_enc.categories_[0]) if dir_enc is not None else ["DR", "CR"]
    channel_cats = list(chan_enc.categories_[0]) if chan_enc is not None else []

    # 2. Extract linear model weights
    clf_step = pipeline.named_steps["classifier"]

    # If wrapped in CalibratedClassifierCV, extract base estimator weights
    if hasattr(clf_step, "calibrated_classifiers_"):
        # Average the weights across calibrated folds
        coefs = []
        intercepts = []
        for cal_clf in clf_step.calibrated_classifiers_:
            base_estimator = getattr(cal_clf, "estimator", getattr(cal_clf, "base_estimator", None))
            if base_estimator is not None and hasattr(base_estimator, "coef_"):
                coefs.append(base_estimator.coef_)
                intercepts.append(base_estimator.intercept_)
        if coefs:
            coef = np.mean(coefs, axis=0).astype(np.float32)
            intercept = np.mean(intercepts, axis=0).astype(np.float32)
        else:
            raise ValueError("Could not extract coef_ from calibrated classifiers")
    elif hasattr(clf_step, "coef_"):
        coef = clf_step.coef_.astype(np.float32)
        intercept = clf_step.intercept_.astype(np.float32)
    else:
        raise ValueError(f"Classifier {clf_step} has no coef_ attribute")

    # Save weights into .npz
    np.savez_compressed(
        output_dir / "classifier_weights.npz",
        coef=coef,
        intercept=intercept,
        classes=np.array(classes),
        direction_categories=np.array(direction_cats),
        channel_categories=np.array(channel_cats),
    )

    # Save metadata
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Exported numpy model artifacts successfully to %s", output_dir)

