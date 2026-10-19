import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def test_training_produces_valid_artifacts(tmp_path):
    subprocess.run(
        [sys.executable, str(PROJECT / "src" / "train.py")],
        cwd=tmp_path,
        check=True,
    )
    artifacts = tmp_path / "artifacts"
    assert (artifacts / "model.joblib").stat().st_size > 0
    metrics = json.loads((artifacts / "metrics.json").read_text())
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1


def test_quality_gate_rejects_bad_model(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "model.joblib").write_bytes(b"placeholder")
    (artifacts / "metrics.json").write_text(json.dumps({
        "precision": 0.9, "recall": 0.2, "roc_auc": 0.9
    }))
    env = os.environ.copy()
    env["MIN_RECALL"] = "0.70"
    result = subprocess.run(
        [sys.executable, str(PROJECT / "src" / "validate_model.py")],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "quality gate failed" in result.stdout + result.stderr
