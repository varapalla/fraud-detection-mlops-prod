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

    try:
        metrics = json.loads(metrics_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"FAIL: invalid metrics.json: {exc}") from exc

    thresholds = {
        "precision": float(os.getenv("MIN_PRECISION", "0.50")),
        "recall": float(os.getenv("MIN_RECALL", "0.70")),
        "roc_auc": float(os.getenv("MIN_ROC_AUC", "0.80")),
    }

    failed = []

    print("Model quality gate report")
    print("---------------------------------------------")

    for name, minimum in thresholds.items():
        actual = metrics.get(name)

        if (
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not 0 <= actual <= 1
        ):
            print(f"{name}: INVALID [FAIL]")
            failed.append(name)
            continue

        status = "PASS" if actual >= minimum else "FAIL"
        print(
            f"{name}: {actual:.4f} "
            f"(minimum {minimum:.4f}) [{status}]"
        )

        if actual < minimum:
            failed.append(name)

    try:
        artifact = joblib.load(model_path)

        if not isinstance(artifact, dict):
            raise ValueError("Expected a dictionary containing model and threshold")

        model = artifact.get("model")
        threshold = artifact.get("threshold")

        if not callable(getattr(model, "predict_proba", None)):
            raise ValueError("Artifact model does not support predict_proba")

        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("Artifact threshold must be between 0 and 1")

        if abs(float(threshold) - float(metrics.get("threshold", -1))) > 1e-9:
            raise ValueError("Threshold differs between model artifact and metrics")

    except Exception as exc:
        raise SystemExit(f"FAIL: invalid model artifact: {exc}") from exc

    if failed:
        raise SystemExit(
            "FAIL: quality gates not met: " + ", ".join(failed)
        )

    print("PASS: model artifact and quality gates validated")


if __name__ == "__main__":
    main()
