# fraud-detection-mlops-prod

Production-oriented MLOps CI/CD lab for a synthetic BFSI fraud-detection use case.

## What this POC demonstrates

- Python linting and unit tests with GitHub Actions
- Synthetic model training and model-quality gates
- Dependency auditing with pip-audit
- Docker image build and Trivy vulnerability scan
- GHCR publishing on trusted pushes to `main`
- FastAPI model inference endpoint, ready for later Kubernetes deployment

## Repository layout

```text
.github/workflows/ml-ci.yaml
src/train.py
src/validate_model.py
src/main.py
tests/test_training.py
requirements.txt
requirements-dev.txt
Dockerfile
```

## Local setup

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
ruff check src tests
pytest -v
python src/train.py
python src/validate_model.py
```

## Run the API

```bash
python src/train.py
uvicorn src.main:app --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/health
```

The `/predict` endpoint expects exactly 20 numeric features. Open
`http://127.0.0.1:8080/docs` for interactive API documentation.

## Build and run the container

Generate the model artifact first, because it is intentionally excluded from Git.

```bash
python src/train.py
docker build -t fraud-detection-mlops-prod:local .
docker run --rm -p 8080:8080 fraud-detection-mlops-prod:local
```

## GitHub setup

1. Create a repository named `fraud-detection-mlops-prod`.
2. Add these files and push to `main`.
3. Open a pull request to run CI.
4. Protect `main` and require successful checks before merging.
5. Ensure GitHub Actions has package write permission for GHCR publishing.

The sample workflow uses action version tags for readability. Before production use,
pin third-party actions to reviewed full commit SHAs and review all dependency updates.
The workflow publishes only after the validation and container scan jobs pass on a
push to `main`.

## Important notes

- The dataset is synthetic; it is not real banking data.
- Precision, recall and ROC-AUC thresholds are demonstration defaults, not approved
  fraud-risk thresholds.
- This is a production-oriented learning POC, not a production-ready fraud decision
  system. Do not use it to make real financial decisions.
- The model artifact is generated during CI and is not committed to Git.
