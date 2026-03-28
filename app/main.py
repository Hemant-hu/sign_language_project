"""
app/main.py
===========
Step 7 & 8 — FastAPI backend exposing the sign language prediction system
as a REST API.

Endpoints
---------
GET  /                    Health check
GET  /info                Model metadata
POST /predict             Predict sign from a JSON sequence
POST /predict/file        Predict sign from an uploaded .npy file
POST /predict/batch       Predict multiple sequences at once
POST /sentence/reset      Reset the sign history (start a new sentence)
GET  /sentence            Get the current accumulated sentence

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import io
import sys
import json
import tempfile
import numpy as np

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any

# ── project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import SignPredictor, CONFIDENCE_THRESHOLD

# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Sign Language to Text API",
    description = "LSTM-based sign language recognition with next-sign prediction",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# Allow all origins for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Lazy-load predictor once at startup ──────────────────────────────────────
_predictor: SignPredictor | None = None


def get_predictor() -> SignPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SignPredictor()
    return _predictor


@app.on_event("startup")
async def startup_event() -> None:
    print("[app] Loading model …")
    get_predictor()
    print("[app] Model ready.")


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────

class SequenceInput(BaseModel):
    """
    Single sequence input.

    sequence : 2-D list  [[f1, f2, …], [f1, f2, …], …]
               or 1-D flat list  [f1, f2, f3, …]
    top_k    : number of top predictions to return (default 3)
    """
    sequence : list[list[float]] | list[float] = Field(
        ...,
        description="Landmark sequence: 2-D (timesteps × features) or flat 1-D list",
        example=[[0.1] * 63] * 20,
    )
    top_k    : int = Field(3, ge=1, le=10, description="Number of top predictions")


class BatchSequenceInput(BaseModel):
    sequences: list[list[list[float]] | list[float]] = Field(
        ..., description="List of sequences"
    )
    top_k    : int = Field(3, ge=1, le=10)


class PredictionResponse(BaseModel):
    current_sign     : str
    confidence       : float
    top_predictions  : list[dict[str, Any]]
    next_prediction  : str | None
    next_suggestions : list[str]
    sentence         : str
    history          : list[str]


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Sign Language to Text API is running."}


@app.get("/info", tags=["Model"])
def model_info():
    """Return model metadata (input shape, classes, accuracy)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta_path = os.path.join(root, "models", "model_meta.json")
    labels_path = os.path.join(root, "models", "labels.pkl")

    meta: dict[str, Any] = {}

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta.update(json.load(f))
    else:
        meta["warning"] = "model_meta.json not found — run training first."

    if os.path.exists(labels_path):
        import pickle
        with open(labels_path, "rb") as f:
            le = pickle.load(f)
        meta["classes"] = list(le.classes_)
    else:
        meta["classes"] = []

    meta["confidence_threshold"] = CONFIDENCE_THRESHOLD
    return meta


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(body: SequenceInput):
    """
    Predict the sign from a landmark sequence.

    **Input**
    ```json
    {
      "sequence": [[0.1, 0.2, …], …],   // shape: (20, 63) recommended
      "top_k": 3
    }
    ```

    **Output**
    ```json
    {
      "current_sign":     "HELLO",
      "confidence":       0.87,
      "top_predictions":  [{"sign": "HELLO", "confidence": 0.87}, …],
      "next_prediction":  "HOW",
      "next_suggestions": ["HOW", "YOU", "WHAT"],
      "sentence":         "Hello how",
      "history":          ["HELLO", "HOW"]
    }
    ```
    """
    try:
        seq = np.array(body.sequence, dtype=np.float32)
        result = get_predictor().predict(seq, top_k=body.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/file", response_model=PredictionResponse, tags=["Prediction"])
async def predict_from_file(file: UploadFile = File(...)):
    """
    Predict from an uploaded .npy file containing a landmark sequence.

    The .npy file should contain an array of shape (T, F) or (1, T, F).
    """
    if not file.filename.endswith(".npy"):
        raise HTTPException(
            status_code=400,
            detail="Only .npy files are accepted."
        )
    contents = await file.read()
    try:
        arr = np.load(io.BytesIO(contents), allow_pickle=True).astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse .npy: {e}")

    try:
        result = get_predictor().predict(arr)
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(body: BatchSequenceInput):
    """
    Predict signs for multiple sequences in one request.
    Returns a list of prediction results.
    Note: batch predictions do NOT update the sign history.
    """
    predictor = get_predictor()
    results   = []
    for raw_seq in body.sequences:
        try:
            seq    = np.array(raw_seq, dtype=np.float32)
            result = predictor.predict(seq, top_k=body.top_k)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})
    return {"predictions": results, "count": len(results)}


@app.post("/sentence/reset", tags=["Sentence"])
def reset_sentence():
    """Clear the sign history and start a new sentence."""
    get_predictor().reset_history()
    return {"message": "History cleared. Ready for a new sentence."}


@app.get("/sentence", tags=["Sentence"])
def get_sentence():
    """Return the sentence built from the accumulated sign history."""
    predictor = get_predictor()
    return {
        "sentence": predictor.get_sentence(),
        "history" : predictor._history,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dev server entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
