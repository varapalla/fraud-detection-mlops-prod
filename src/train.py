
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT = Path("artifacts")

MIN_PRECISION = 0.50
MIN_RECALL = 0.70


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Generate a reproducible, synthetic imbalanced dataset.
    X, y = make_classification(
        n_samples=3000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.95, 0.05],
        class_sep=1.2,
        random_state=42,
    )

    # Split into training (60%), validation (20%), and test (20%).
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )

    # Train the model only on the training dataset.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    )
    model.fit(X_train, y_train)

    # Tune the classification threshold using validation data only.
    val_probabilities = model.predict_proba(X_val)[:, 1]
    candidates = np.arange(0.05, 0.96, 0.01)

    best_threshold = None
    best_precision = -1.0
    best_recall = 0.0

    for threshold in candidates:
        val_predictions = (val_probabilities >= threshold).astype(int)

        precision = precision_score(
            y_val, val_predictions, zero_division=0
        )
        recall = recall_score(
            y_val, val_predictions, zero_division=0
        )

        # Choose the highest precision among thresholds meeting recall.
        if recall >= MIN_RECALL and precision > best_precision:
            best_threshold = float(threshold)
            best_precision = float(precision)
            best_recall = float(recall)

    if best_threshold is None:
        raise SystemExit(
            "No validation threshold met the minimum recall requirement. "
            "Review the model or quality requirements."
        )

    print(
        f"Selected threshold: {best_threshold:.2f}; "
        f"validation precision: {best_precision:.4f}; "
        f"validation recall: {best_recall:.4f}"
    )

    # Evaluate exactly once on the held-out test dataset.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= best_threshold).astype(int)

    metrics = {
        "precision": float(
            precision_score(y_test, test_predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, test_predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "validation_precision": best_precision,
        "validation_recall": best_recall,
        "threshold": best_threshold,
        "model_type": "LogisticRegression",
        "dataset": "synthetic-fraud-data",
    }

    # Save the model and its decision threshold together.
    joblib.dump(
        {"model": model, "threshold": best_threshold},
        OUTPUT / "model.joblib",
    )
    (OUTPUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
