"""numpy-only ML model inference for serverless deployment without scikit-learn/scipy/pandas."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import numpy as np

from .features import TransactionFeatures

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "v1"


@dataclass
class MLPrediction:
    """Prediction result from numpy ML model."""
    category: str
    confidence: float
    all_probabilities: dict[str, float] = field(default_factory=dict)
    model_version: str = "v1"


class NumpyClassifier:
    """Portable numpy-only inference engine executing TF-IDF + Calibrated Linear Model."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self._vocab: dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._char_vocab: dict[str, int] = {}
        self._char_idf: Optional[np.ndarray] = None
        self._direction_map: dict[str, int] = {}
        self._channel_map: dict[str, int] = {}
        self._coef: Optional[np.ndarray] = None
        self._intercept: Optional[np.ndarray] = None
        self._classes: list[str] = []
        self._metadata: dict[str, Any] = {}
        self._loaded: bool = False

        self.load()

    def load(self) -> bool:
        """Load model weights and vocabulary files from model directory."""
        if not self.model_dir.is_dir():
            logger.info("Model directory not found at %s. ML inference unavailable until trained.", self.model_dir)
            return False

        try:
            vocab_path = self.model_dir / "tfidf_vocab.json"
            idf_path = self.model_dir / "tfidf_idf.npy"
            char_vocab_path = self.model_dir / "char_tfidf_vocab.json"
            char_idf_path = self.model_dir / "char_tfidf_idf.npy"
            weights_path = self.model_dir / "classifier_weights.npz"
            meta_path = self.model_dir / "metadata.json"

            if not (vocab_path.is_file() and idf_path.is_file() and weights_path.is_file()):
                logger.info("Model weights not yet generated in %s", self.model_dir)
                return False

            with open(vocab_path, "r", encoding="utf-8") as f:
                self._vocab = json.load(f)
            self._idf = np.load(idf_path)

            if char_vocab_path.is_file() and char_idf_path.is_file():
                with open(char_vocab_path, "r", encoding="utf-8") as f:
                    self._char_vocab = json.load(f)
                self._char_idf = np.load(char_idf_path)

            weights = np.load(weights_path, allow_pickle=True)
            self._coef = weights["coef"]
            self._intercept = weights["intercept"]
            self._classes = list(weights["classes"])

            if "direction_categories" in weights:
                dirs = list(weights["direction_categories"])
                self._direction_map = {d: i for i, d in enumerate(dirs)}
            if "channel_categories" in weights:
                chans = list(weights["channel_categories"])
                self._channel_map = {c: i for i, c in enumerate(chans)}

            if meta_path.is_file():
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)

            self._loaded = True
            logger.info("Loaded numpy classifier: %d classes, %d word features.", len(self._classes), len(self._vocab))
            return True
        except Exception as e:
            logger.error("Failed to load numpy classifier artifacts from %s: %s", self.model_dir, e)
            self._loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _extract_word_ngrams(self, text: str) -> list[str]:
        words = re.findall(r"(?u)\b\w+\b", text.lower())
        tokens = list(words)
        # Add bigrams
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]} {words[i+1]}")
        return tokens

    def _extract_char_ngrams(self, text: str, min_n: int = 3, max_n: int = 4) -> list[str]:
        tokens = []
        words = re.findall(r"(?u)\b\w+\b", text.lower())
        for w in words:
            w_bound = f" {w} "
            for n in range(min_n, max_n + 1):
                for i in range(len(w_bound) - n + 1):
                    tokens.append(w_bound[i : i + n])
        return tokens

    def _tfidf_transform(
        self,
        tokens: list[str],
        vocab: dict[str, int],
        idf: np.ndarray,
        sublinear_tf: bool = True
    ) -> np.ndarray:
        vec = np.zeros(len(vocab), dtype=np.float32)
        if not tokens:
            return vec

        counts: dict[int, int] = {}
        for t in tokens:
            idx = vocab.get(t)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1

        for idx, count in counts.items():
            tf = (1.0 + math.log(count)) if sublinear_tf else float(count)
            vec[idx] = tf * idf[idx]

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def predict(self, features: TransactionFeatures) -> MLPrediction:
        """Predict transaction category using pure numpy vector arithmetic."""
        if not self._loaded or self._coef is None or self._intercept is None:
            return MLPrediction(
                category="Other/Uncategorized",
                confidence=0.0,
                all_probabilities={},
                model_version="untrained"
            )

        # 1. Word TF-IDF
        desc_text = f"{features.description_cleaned} {features.counterparty}".strip()
        word_tokens = self._extract_word_ngrams(desc_text)
        word_vec = self._tfidf_transform(word_tokens, self._vocab, self._idf)

        feature_blocks = [word_vec]

        # 2. Char TF-IDF if present
        if self._char_vocab and self._char_idf is not None:
            char_tokens = self._extract_char_ngrams(desc_text)
            char_vec = self._tfidf_transform(char_tokens, self._char_vocab, self._char_idf)
            feature_blocks.append(char_vec)

        # 3. Direction One-Hot
        if self._direction_map:
            dir_vec = np.zeros(len(self._direction_map), dtype=np.float32)
            d_idx = self._direction_map.get(features.direction.upper())
            if d_idx is not None:
                dir_vec[d_idx] = 1.0
            feature_blocks.append(dir_vec)

        # 4. Channel One-Hot
        if self._channel_map:
            chan_vec = np.zeros(len(self._channel_map), dtype=np.float32)
            c_idx = self._channel_map.get(features.payment_channel.upper())
            if c_idx is not None:
                chan_vec[c_idx] = 1.0
            feature_blocks.append(chan_vec)

        # 5. Log amount feature
        amt_float = float(features.amount) if features.amount else 0.0
        log_amt = np.array([math.log1p(max(0.0, amt_float))], dtype=np.float32)
        feature_blocks.append(log_amt)

        full_features = np.concatenate(feature_blocks)

        # Match dimensions if needed (safety truncation/padding)
        expected_len = self._coef.shape[1]
        if len(full_features) != expected_len:
            if len(full_features) > expected_len:
                full_features = full_features[:expected_len]
            else:
                padded = np.zeros(expected_len, dtype=np.float32)
                padded[: len(full_features)] = full_features
                full_features = padded

        # 6. Linear prediction: logits = W @ x + b
        logits = np.dot(self._coef, full_features) + self._intercept

        # 7. Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # 8. Top prediction
        top_idx = int(np.argmax(probs))
        top_category = self._classes[top_idx]
        top_conf = float(probs[top_idx])

        prob_dict = {
            self._classes[i]: float(probs[i])
            for i in range(len(self._classes))
        }

        return MLPrediction(
            category=top_category,
            confidence=top_conf,
            all_probabilities=prob_dict,
            model_version=self._metadata.get("version", "v1")
        )

