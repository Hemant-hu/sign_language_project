"""
predict.py
==========
Step 5 — Load the trained model and label encoder, accept an input sequence,
and return the predicted sign plus next-sign suggestions.

Can be used as:
  • an imported module by the FastAPI app
  • a standalone CLI tool for quick testing

CLI usage:
    python src/predict.py                      # random synthetic test
    python src/predict.py path/to/seq.npy      # predict from .npy file
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train            import load_trained_model, MODEL_PATH
from src.preprocess       import load_label_encoder, LABELS_PATH, normalize_X
from src.sequence_predictor import BigramPredictor, signs_to_sentence

# ──────────────────────────────────────────────────────────────────────────────
# Confidence threshold — predictions below this are returned as "UNCERTAIN"
# ──────────────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.40


# ──────────────────────────────────────────────────────────────────────────────
# Predictor class  (stateful: caches model, encoder, and bigram)
# ──────────────────────────────────────────────────────────────────────────────

class SignPredictor:
    """
    Stateful prediction wrapper.

    Maintains an internal history of recently predicted signs to enable
    sentence assembly and next-sign prediction.
    """

    def __init__(
        self,
        model_path : str = MODEL_PATH,
        labels_path: str = LABELS_PATH,
    ) -> None:
        self.model   = load_trained_model(model_path)
        self.encoder = load_label_encoder(labels_path)
        self.bigram  = BigramPredictor.load()
        self._history: list[str] = []

        # Load model metadata for shape info
        meta_path = os.path.join(os.path.dirname(model_path), "model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self._meta = json.load(f)
        else:
            self._meta = {}

    # ── Core prediction ──────────────────────────────────────────────────────

    def predict(
        self,
        sequence: np.ndarray | list,
        top_k: int = 3,
    ) -> dict:
        """
        Predict the sign from a single input sequence.

        Parameters
        ----------
        sequence : array-like, shape (T, F) or (1, T, F) or flat list
            The landmark sequence to classify.
        top_k : int
            How many top predictions to include in the response.

        Returns
        -------
        dict with keys:
            current_sign   : str
            confidence     : float  (0–1)
            top_predictions: list[dict]  (sign, confidence)
            next_prediction: str
            next_suggestions: list[str]
            sentence       : str
            history        : list[str]
        """
        seq = self._prepare_input(sequence)

        # ── Model inference ──────────────────────────────────────────────────
        probs     = self.model.predict(seq, verbose=0)[0]  # (num_classes,)
        top_idx   = np.argsort(probs)[::-1][:top_k]

        best_idx  = int(top_idx[0])
        best_prob = float(probs[best_idx])
        best_sign = (
            self.encoder.classes_[best_idx]
            if best_prob >= CONFIDENCE_THRESHOLD
            else "UNCERTAIN"
        )

        top_preds = [
            {"sign": self.encoder.classes_[i], "confidence": round(float(probs[i]), 4)}
            for i in top_idx
        ]

        # ── History & next-sign ──────────────────────────────────────────────
        if best_sign != "UNCERTAIN":
            self._history.append(best_sign)

        next_suggestions = self.bigram.predict_from_history(self._history, n=3)
        next_sign        = next_suggestions[0] if next_suggestions else None

        sentence = signs_to_sentence(self._history)

        return {
            "current_sign"   : best_sign,
            "confidence"     : round(best_prob, 4),
            "top_predictions": top_preds,
            "next_prediction": next_sign,
            "next_suggestions": next_suggestions,
            "sentence"       : sentence,
            "history"        : list(self._history),
        }

    # ── History helpers ──────────────────────────────────────────────────────

    def reset_history(self) -> None:
        """Clear the accumulated sign history (start a new sentence)."""
        self._history.clear()

    def get_sentence(self) -> str:
        return signs_to_sentence(self._history)

    # ── Input preparation ────────────────────────────────────────────────────

    def _prepare_input(self, sequence: np.ndarray | list) -> np.ndarray:
        """
        Accept various input shapes and return a (1, T, F) float32 array
        correctly normalised for the model.
        """
        arr = np.array(sequence, dtype=np.float32)

        if arr.ndim == 1:
            # Flat vector: (T*F,) → (T, F)
            seq_len = self._meta.get("sequence_length", 20)
            n_feat  = self._meta.get("num_features", arr.shape[0] // seq_len)
            arr = arr.reshape(seq_len, n_feat)

        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]    # → (1, T, F)

        # Normalise (same as training)
        arr = normalize_X(arr)
        return arr


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience function
# ──────────────────────────────────────────────────────────────────────────────

_predictor: SignPredictor | None = None


def get_predictor() -> SignPredictor:
    """Return a cached SignPredictor (lazy initialisation)."""
    global _predictor
    if _predictor is None:
        _predictor = SignPredictor()
    return _predictor


def predict_sequence(sequence: np.ndarray | list) -> dict:
    """Single-call convenience wrapper used by the FastAPI app."""
    return get_predictor().predict(sequence)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = SignPredictor()

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        # Load sequence from file
        raw = np.load(sys.argv[1], allow_pickle=True).astype(np.float32)
        print(f"[predict] Loaded sequence from {sys.argv[1]}  shape={raw.shape}")
    else:
        # Generate a random synthetic sequence for smoke-testing
        meta_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "model_meta.json"
        )
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            T, F = meta["sequence_length"], meta["num_features"]
        else:
            T, F = 20, 63
        raw = np.random.rand(T, F).astype(np.float32)
        print(f"[predict] Using synthetic sequence  shape=({T}, {F})")

    result = predictor.predict(raw)

    print("\n" + "═" * 45)
    print("  PREDICTION RESULT")
    print("═" * 45)
    print(f"  Current sign    : {result['current_sign']}")
    print(f"  Confidence      : {result['confidence']:.1%}")
    print(f"  Next prediction : {result['next_prediction']}")
    print(f"  Sentence so far : \"{result['sentence']}\"")
    print(f"  History         : {result['history']}")
    print("\n  Top predictions :")
    for p in result["top_predictions"]:
        bar = "█" * int(p["confidence"] * 20)
        print(f"    {p['sign']:>20}  {p['confidence']:.1%}  {bar}")
    print("═" * 45)
