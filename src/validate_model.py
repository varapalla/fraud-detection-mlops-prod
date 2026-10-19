import json
import os
from pathlib import Path

import joblib

ARTIFACTS = Path("artifacts")


def main():
    model_path = ARTIFACTS / "model.joblib"
    metrics_path = ARTIFACTS / "metrics.json"
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise SystemExit("FAIL: model artifact is missing or empty")
    if not metrics_path.is_file():
        raise SystemExit("FAIL: metrics.json is missing")

    metrics = json.loads(metrics_path.read_text())
    for name in ("precision", "recall", "roc_auc"):
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise SystemExit(f"FAIL: invalid metric: {name}")

    thresholds = {
        "precision": float(os.getenv("MIN_PRECISION", "0.50")),
        "recall": float(os.getenv("MIN_RECALL", "0.70")),
        "roc_auc": float(os.getenv("MIN_ROC_AUC", "0.80")),
    }
    for name, minimum in thresholds.items():
        actual = metrics[name]
        print(f"{name}: {actual:.4f} (minimum {minimum:.4f})")
        if actual < minimum:
            raise SystemExit(f"FAIL: {name} quality gate failed")

    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise SystemExit("FAIL: artifact is not a prediction model")
    print("PASS: model artifact and quality gates validated")


if __name__ == "__main__":
    main()
