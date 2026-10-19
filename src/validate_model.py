
import json
import math
import os
import sys
from pathlib import Path

import joblib

ARTIFACTS = Path("artifacts")


def read_minimum(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default))

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a number between 0 and 1"
        ) from exc

    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")

    return value


def main() -> int:
    model_path = ARTIFACTS / "model.joblib"
    metrics_path = ARTIFACTS / "metrics.json"

    if not model_path.is_file() or model_path.stat().st_size == 0:
        print(
            "FAIL: model artifact is missing or empty",
            file=sys.stderr,
        )
        return 1

    if not metrics_path.is_file():
        print("FAIL: metrics.json is missing", file=sys.stderr)
        return 1

    try:
        metrics = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: invalid metrics.json: {exc}", file=sys.stderr)
        return 1

    if not isinstance(metrics, dict):
        print(
            "FAIL: metrics.json must contain a JSON object",
            file=sys.stderr,
        )
        return 1

    try:
        minimums = {
            "precision": read_minimum("MIN_PRECISION", 0.50),
            "recall": read_minimum("MIN_RECALL", 0.70),
            "roc_auc": read_minimum("MIN_ROC_AUC", 0.80),
        }
    except ValueError as exc:
        print(f"FAIL: invalid quality-gate configuration: {exc}", file=sys.stderr)
        return 1

    failed = []

    print("Model quality gate report")
    print("---------------------------------------------")

    for name, minimum in minimums.items():
        actual = metrics.get(name)

        valid = (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(actual)
            and 0.0 <= actual <= 1.0
        )

        if not valid:
            print(f"{name}: INVALID [FAIL]")
            failed.append(name)
            continue

        passed = actual >= minimum
        status = "PASS" if passed else "FAIL"

        print(
            f"{name}: {actual:.4f} "
            f"(minimum {minimum:.4f}) [{status}]"
        )

        if not passed:
            failed.append(name)

    # Reject poor metrics before attempting to load the model.
    if failed:
        print(
            "FAIL: quality gate failed: " + ", ".join(failed),
            file=sys.stderr,
        )
        return 1

    try:
        artifact = joblib.load(model_path)

        if not isinstance(artifact, dict):
            raise ValueError(
                "Expected a dictionary containing model and threshold"
            )

        model = artifact.get("model")
        threshold = artifact.get("threshold")

        if not callable(getattr(model, "predict_proba", None)):
            raise ValueError("Model does not support predict_proba")

        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError("Invalid model decision threshold")

        metrics_threshold = metrics.get("threshold")

        if (
            not isinstance(metrics_threshold, (int, float))
            or isinstance(metrics_threshold, bool)
            or not math.isfinite(metrics_threshold)
            or abs(float(threshold) - float(metrics_threshold)) > 1e-9
        ):
            raise ValueError(
                "Model threshold does not match metrics.json"
            )

    except Exception as exc:
        print(f"FAIL: invalid model artifact: {exc}", file=sys.stderr)
        return 1

    print("PASS: model artifact and quality gates validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
