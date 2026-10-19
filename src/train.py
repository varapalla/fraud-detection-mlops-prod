import json
from pathlib import Path

import joblib
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT = Path("artifacts")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    X, y = make_classification(
        n_samples=3000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.95, 0.05],
        class_sep=1.2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "model_type": "LogisticRegression",
        "dataset": "synthetic-fraud-data",
    }
    joblib.dump(model, OUTPUT / "model.joblib")
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
