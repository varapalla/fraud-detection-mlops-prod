import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

OUTPUT = Path("artifacts")
MIN_VALIDATION_RECALL = 0.70


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    X, y = make_classification(
        n_samples=3000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.95, 0.05],
        class_sep=1.8,
        random_state=42,
    )

    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.50,
        stratify=y_remaining,
        random_state=42,
    )

    model = ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)[:, 1]
    candidates = []

    for threshold in np.arange(0.01, 0.96, 0.01):
        predictions = val_probabilities >= threshold
        precision = precision_score(
            y_val, predictions, zero_division=0
        )
        recall = recall_score(
            y_val, predictions, zero_division=0
        )

        if recall >= MIN_VALIDATION_RECALL:
            candidates.append(
                (float(precision), float(recall), float(threshold))
            )

    if not candidates:
        raise SystemExit(
            "FAIL: no threshold meets validation recall >= 0.70"
        )

    best_precision, best_recall, best_threshold = max(
        candidates, key=lambda item: (item[0], item[1])
    )

    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = test_probabilities >= best_threshold

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
        "model_type": "ExtraTreesClassifier",
        "dataset": "synthetic-fraud-data",
    }

    joblib.dump(
        {"model": model, "threshold": best_threshold},
        OUTPUT / "model.joblib",
    )
    (OUTPUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    print(f"Selected threshold: {best_threshold:.2f}")
    print("Test evaluation metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
