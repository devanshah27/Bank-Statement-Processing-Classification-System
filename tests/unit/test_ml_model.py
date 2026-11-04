"""Unit tests for portable numpy ML classifier."""

from decimal import Decimal
import pytest

from core.classification.features import extract_features
from core.classification.ml_model import NumpyClassifier


@pytest.fixture
def classifier():
    clf = NumpyClassifier()
    return clf


def test_numpy_classifier_loaded(classifier):
    assert classifier.is_loaded is True
    assert len(classifier._classes) == 23
    assert len(classifier._vocab) > 100


def test_numpy_classifier_predict_valid_category(classifier):
    feat = extract_features("ZEPTO GROCERY DELIVERY SERVICES", "DR", Decimal("650.00"))
    pred = classifier.predict(feat)
    assert pred.category in classifier._classes
    assert 0.0 <= pred.confidence <= 1.0


def test_numpy_classifier_probabilities_sum_to_one(classifier):
    feat = extract_features("RELIANCE DIGITAL ELECTRONICS STORE", "DR", Decimal("12500.00"))
    pred = classifier.predict(feat)
    total_prob = sum(pred.all_probabilities.values())
    assert abs(total_prob - 1.0) < 0.01

