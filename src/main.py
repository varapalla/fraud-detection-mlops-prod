from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("artifacts/model.joblib")
if not MODEL_PATH.is_file():
    raise RuntimeError("Model artifact not found. Run src/train.py first.")

model = joblib.load(MODEL_PATH)
app = FastAPI(title="Fraud Detection API", version="1.0.0")


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=20, max_length=20)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        probability = float(model.predict_proba([request.features])[0][1])
        return {
            "fraud_probability": probability,
            "predicted_fraud": probability >= 0.5,
        }
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="Invalid feature values") from exc
