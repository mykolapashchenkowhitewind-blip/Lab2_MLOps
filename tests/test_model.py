from pathlib import Path

import joblib

from ml.train import train_and_save


def test_train_creates_model_file(tmp_path: Path):
    model_file = tmp_path / "model.joblib"

    accuracy = train_and_save(model_path=model_file)

    assert model_file.exists(), "Model file should be created"
    assert 0.0 <= accuracy <= 1.0, "Accuracy should be a valid ratio"
    assert accuracy > 0.8, f"Expected accuracy > 0.8, got {accuracy}"


def test_model_predicts_three_classes(tmp_path: Path):
    model_file = tmp_path / "model.joblib"
    train_and_save(model_path=model_file)
    model = joblib.load(model_file)

    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])

    assert prediction[0] in (0, 1, 2), "Class should be one of 0, 1, or 2"
