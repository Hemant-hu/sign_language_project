"""
preprocess.py
=============
Step 2 — Normalise features, encode labels, and produce train/test splits
ready for model training.

Run standalone:
    python src/preprocess.py [optional_dataset_root]
"""

from __future__ import annotations

import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

# ── project root on sys.path ──────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.load_data import load_dataset

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalize_X(X: np.ndarray) -> np.ndarray:
    """
    Per-feature min-max normalisation across the entire dataset.

    Scales each feature dimension to [0, 1] using the global min/max.
    This keeps relative motion patterns intact while bringing all
    landmark coordinates to the same numeric range.

    Parameters
    ----------
    X : ndarray, shape (N, T, F)

    Returns
    -------
    X_norm : ndarray, shape (N, T, F), float32
    """
    X = X.astype(np.float32)
    # Reshape to (N*T, F) to compute per-feature statistics
    N, T, F = X.shape
    flat    = X.reshape(-1, F)

    x_min = flat.min(axis=0)          # (F,)
    x_max = flat.max(axis=0)          # (F,)
    denom = (x_max - x_min)
    denom[denom < 1e-8] = 1.0         # avoid division by zero for static dims

    X_norm = (flat - x_min) / denom
    return X_norm.reshape(N, T, F).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Label encoding
# ──────────────────────────────────────────────────────────────────────────────

def encode_labels(y: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    """
    Fit a sklearn LabelEncoder on y (string class names) and return
    (y_encoded_as_int, encoder).

    The encoder is later used to:
      - convert integer predictions back to class names
      - build the one-hot matrix for training
    """
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    print(f"[preprocess] {len(le.classes_)} classes encoded.")
    return y_int.astype(np.int32), le


def save_label_encoder(le: LabelEncoder, path: str = LABELS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(le, f)
    print(f"[preprocess] LabelEncoder saved → {path}")


def load_label_encoder(path: str = LABELS_PATH) -> LabelEncoder:
    with open(path, "rb") as f:
        le = pickle.load(f)
    print(f"[preprocess] LabelEncoder loaded from {path}")
    return le


# ──────────────────────────────────────────────────────────────────────────────
# 3.  One-hot conversion
# ──────────────────────────────────────────────────────────────────────────────

def to_categorical(y_int: np.ndarray, num_classes: int) -> np.ndarray:
    return tf.keras.utils.to_categorical(y_int, num_classes=num_classes)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Train / test split
# ──────────────────────────────────────────────────────────────────────────────

def split_data(
    X: np.ndarray,
    y_onehot: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple:
    """
    Stratified 80/20 train-test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    # Stratify on argmax of one-hot labels
    y_int = np.argmax(y_onehot, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot,
        test_size=test_size,
        random_state=random_state,
        stratify=y_int,
    )
    print(f"[preprocess] Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Master pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_preprocessing(
    dataset_root: str | None = None,
    test_size: float = 0.20,
) -> dict:
    """
    Full preprocessing pipeline.

    Returns a dict with keys:
        X_train, X_test, y_train, y_test,
        label_encoder, num_classes,
        sequence_length, num_features
    """
    # ── Load raw data ────────────────────────────────────────────────────────
    X, y_raw = load_dataset(dataset_root)

    # ── Normalise ────────────────────────────────────────────────────────────
    print("[preprocess] Normalising features …")
    X = normalize_X(X)

    # ── Encode labels ────────────────────────────────────────────────────────
    y_int, le = encode_labels(y_raw)
    num_classes = len(le.classes_)
    y_onehot = to_categorical(y_int, num_classes)

    # ── Persist encoder ──────────────────────────────────────────────────────
    save_label_encoder(le)

    # ── Split ────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y_onehot, test_size)

    _, sequence_length, num_features = X.shape

    print(f"\n[preprocess] Preprocessing complete.")
    print(f"  Sequence length : {sequence_length}")
    print(f"  Feature dims    : {num_features}")
    print(f"  Classes         : {num_classes}")

    return dict(
        X_train        = X_train,
        X_test         = X_test,
        y_train        = y_train,
        y_test         = y_test,
        label_encoder  = le,
        num_classes    = num_classes,
        sequence_length= sequence_length,
        num_features   = num_features,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else None
    out  = run_preprocessing(root)
    print(f"\nX_train : {out['X_train'].shape}")
    print(f"X_test  : {out['X_test'].shape}")
    print(f"Classes : {list(out['label_encoder'].classes_[:10])} …")
