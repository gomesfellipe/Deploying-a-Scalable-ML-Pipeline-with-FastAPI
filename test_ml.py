from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, train_model

def test_process_data_returns_expected_types():
    """Validate output types from process_data."""
    data_path = Path(__file__).resolve().parent / "data" / "census.csv"
    data = pd.read_csv(data_path)
    train, _ = train_test_split(data, test_size=0.2, random_state=42)

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
        label="salary",
        training=True,
    )

    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert X_train.size > 0
    assert y_train.size > 0
    assert encoder is not None
    assert lb is not None


def test_model_uses_logistic_regression():
    """Ensure the training pipeline uses Logistic Regression."""
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    y = np.array([0, 1, 1, 0])
    model = train_model(X, y)

    classifier = model.named_steps.get("classifier")
    assert classifier is not None
    assert classifier.__class__.__name__ == "LogisticRegression"


def test_compute_metrics_expected_values():
    """Validate compute_model_metrics returns expected values."""
    y_true = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, preds)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 * (precision * recall) / (precision + recall))


def test_train_test_split_sizes():
    """Check train/test split sizes are as expected."""
    data_path = Path(__file__).resolve().parent / "data" / "census.csv"
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    assert len(train) + len(test) == len(data)
    assert len(test) == pytest.approx(len(data) * 0.2, rel=0.02)
