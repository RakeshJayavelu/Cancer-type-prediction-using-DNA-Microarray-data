import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.predict import predict, predict_proba

app = FastAPI(title="Cancer Classifier API", version="1.0")

LABEL_MAP = {0: "ALL", 1: "AML"}


class PredictRequest(BaseModel):
    features: List[float]
    model_name: str = "gaussian"


class PredictResponse(BaseModel):
    prediction: int
    label: str
    probabilities: List[float] = []


@app.get("/")
def root():
    return {"message": "Cancer Classifier API is running"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    try:
        input_array = np.array(request.features).reshape(1, -1)
        prediction = predict(input_array, model_name=request.model_name)
        proba = predict_proba(input_array, model_name=request.model_name)

        pred_int = int(prediction[0])
        probabilities = proba[0].tolist() if proba is not None else []

        return PredictResponse(
            prediction=pred_int,
            label=LABEL_MAP[pred_int],
            probabilities=probabilities,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
