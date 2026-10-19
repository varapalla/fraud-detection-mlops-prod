
import json
import os
import sys
from pathlib import Path

import joblib

ARTIFACTS = Path("artifacts")
REQUIRED_METRICS = ("precision", "recall", "roc_auc")


def load_threshold(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        raise SystemExit(f"FAIL: invalid threshold configuration: {name}")

    if not 0 <= value <= 1:
        raise SystemExit(f"FAIL: threshold must be between 0 and 1: {name}")

    return value


def main() -> None:
    model_path = ARTIFACTS / "model.joblib"
    metrics_path = ARTIFACTS / "metrics.json"

    # 1. Validate artifacts
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise SystemExit("FAIL: model artifact is missing or empty")

    if not metrics_path.is_file():
        raise SystemExit("FAIL: metrics.json is missing")

    # 2. Load and validate metrics
    try:
        metrics = json.loads(metrics_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"FAIL: cannot read metrics.json: {exc}")

    if not isinstance(metrics, dict):
        raise SystemExit("FAIL: metrics.json must contain a JSON object")

    errors = []

    for name in REQUIRED_METRICS:
        value = metrics.get(name)

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
        ):
            errors.append(f"Invalid or missing metric: {name}")

    if errors:
        raise SystemExit("FAIL: " + "; ".join(errors))

    # 3. Enforce configurable quality gates
    thresholds = {
        "precision": load_threshold("MIN_PRECISION", 0.50),
        "recall": load_threshold("MIN_RECALL", 0.70),
        "roc_auc": load_threshold("MIN_ROC_AUC", 0.80),
    }

    failures = []

    print("Model quality gate report")
    print("-" * 45)

    for name, minimum in thresholds.items():
        actual = metrics[name]
        status = "PASS" if actual >= minimum else "FAIL"

        print(
            f"{name}: {actual:.4f} "
            f"(minimum {minimum:.4f}) [{status}]"
        )

        if actual < minimum:
            failures.append(name)

    if failures:
        raise SystemExit(
            "FAIL: quality gates not met: " + ", ".join(failures)
        )

    # 4. Validate model artifact
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise SystemExit(f"FAIL: cannot load model artifact: {exc}")

    if not callable(getattr(model, "predict", None)):
        raise SystemExit("FAIL: artifact is not a prediction model")

    print("PASS: model artifact and quality gates validated")


if __name__ == "__main__":
    main()
