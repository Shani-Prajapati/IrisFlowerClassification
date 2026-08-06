"""
CodeAlpha - Task 1: Iris Flower Classification
FastAPI backend that serves the trained SVM model for predictions.
Run with: uvicorn app:app --reload
"""

import json

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Iris Flower Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("iris_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
with open("model_metadata.json") as f:
    METADATA = json.load(f)

# Species -> display info (used by the UI)
SPECIES_INFO = {
    "Iris-setosa": {"emoji": "🌼", "color": "#7C3AED", "common_name": "Setosa"},
    "Iris-versicolor": {"emoji": "🌸", "color": "#2563EB", "common_name": "Versicolor"},
    "Iris-virginica": {"emoji": "🌺", "color": "#DB2777", "common_name": "Virginica"},
}


class FlowerMeasurements(BaseModel):
    sepal_length: float = Field(..., ge=0, le=15, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=15, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, le=15, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=15, description="Petal width in cm")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_metadata": METADATA}


@app.post("/api/predict")
def predict(measurements: FlowerMeasurements):
    try:
        X = np.array(
            [[
                measurements.sepal_length,
                measurements.sepal_width,
                measurements.petal_length,
                measurements.petal_width,
            ]]
        )
        pred_idx = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        species = label_encoder.inverse_transform([pred_idx])[0]

        probabilities = {
            label_encoder.inverse_transform([i])[0]: round(float(p) * 100, 2)
            for i, p in enumerate(proba)
        }

        info = SPECIES_INFO.get(species, {})
        return {
            "species": species,
            "common_name": info.get("common_name", species),
            "emoji": info.get("emoji", "🌷"),
            "color": info.get("color", "#22C55E"),
            "confidence": round(float(max(proba)) * 100, 2),
            "probabilities": probabilities,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Serve the simple frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
