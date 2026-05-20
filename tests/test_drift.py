import numpy as np
import pytest

from app.drift import DriftDetector


FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def test_no_drift_on_same_distribution():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    current = rng.normal(loc=5.0, scale=1.0, size=(500, 4))

    detector = DriftDetector(reference, FEATURE_NAMES)
    result = detector.detect(current, alpha=0.001)

    assert result["drift_detected"] is False
    assert result["n_drifted_features"] == 0


def test_drift_on_shifted_distribution():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    current = rng.normal(loc=8.0, scale=1.0, size=(500, 4))

    detector = DriftDetector(reference, FEATURE_NAMES)
    result = detector.detect(current, alpha=0.05)

    assert result["drift_detected"] is True
    assert result["n_drifted_features"] == 4
    for feature in FEATURE_NAMES:
        assert result["per_feature"][feature]["p_value"] < 0.05


def test_invalid_current_shape_raises_error():
    reference = np.ones((20, 4))
    current = np.ones((20, 3))

    detector = DriftDetector(reference, FEATURE_NAMES)

    with pytest.raises(ValueError, match="current must be 2D"):
        detector.detect(current)
