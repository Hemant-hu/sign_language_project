"""
train.py
========
Steps 3 & 4 — Build the LSTM model, train it, evaluate it, and save artefacts.

Run:
    python src/train.py
    python src/train.py /path/to/dataset_root
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import run_preprocessing

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "sign_model.h5")
HISTORY_PATH= os.path.join(MODELS_DIR, "training_history.json")
PLOT_PATH   = os.path.join(MODELS_DIR, "training_curves.png")
META_PATH   = os.path.join(MODELS_DIR, "model_meta.json")

os.makedirs(MODELS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Model definition
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    sequence_length: int,
    num_features: int,
    num_classes: int,
    lstm1_units: int = 128,
    lstm2_units: int = 64,
    dense_units: int = 64,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    LSTM sequence classifier.

    Architecture
    ────────────
    Input  →  LSTM(128, return_sequences=True)
           →  LSTM(64)
           →  Dense(64, relu)
           →  Dropout(0.3)
           →  Dense(num_classes, softmax)

    The first LSTM passes its full output sequence to the second LSTM,
    allowing the model to learn temporal dependencies at multiple resolutions.
    """
    inp = keras.Input(shape=(sequence_length, num_features), name="sequence_input")

    x = layers.LSTM(lstm1_units, return_sequences=True, name="lstm_1")(inp)
    x = layers.LSTM(lstm2_units, return_sequences=False, name="lstm_2")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="SignLSTM")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Training
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    dataset_root: str | None = None,
    epochs: int = 30,
    batch_size: int = 32,
    val_split: float = 0.15,
    patience: int = 7,
) -> dict:
    """
    Full training pipeline.

    1. Preprocess data
    2. Build model
    3. Train with early stopping + LR reduction
    4. Evaluate on test set
    5. Save model + artefacts

    Returns the Keras History object's dict.
    """
    # ── Data ─────────────────────────────────────────────────────────────────
    data = run_preprocessing(dataset_root)
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    seq_len = data["sequence_length"]
    n_feat  = data["num_features"]
    n_cls   = data["num_classes"]

    print(f"\n[train] Input  : ({seq_len}, {n_feat})")
    print(f"[train] Classes: {n_cls}")
    print(f"[train] Train  : {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}\n")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(seq_len, n_feat, n_cls)
    model.summary()

    # ── Callbacks ────────────────────────────────────────────────────────────
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Fit ──────────────────────────────────────────────────────────────────
    print("\n[train] Training started …\n")
    history = model.fit(
        X_train, y_train,
        epochs          = epochs,
        batch_size      = batch_size,
        validation_split= val_split,
        callbacks       = cbs,
        verbose         = 1,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n[train] Evaluating on test set …")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test loss     : {test_loss:.4f}")
    print(f"  Test accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")

    # ── Save model ───────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n[train] Model saved → {MODEL_PATH}")

    # ── Save metadata ────────────────────────────────────────────────────────
    meta = {
        "sequence_length": seq_len,
        "num_features"   : n_feat,
        "num_classes"    : n_cls,
        "test_accuracy"  : float(test_acc),
        "test_loss"      : float(test_loss),
        "epochs_trained" : len(history.history["loss"]),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] Metadata saved → {META_PATH}")

    # ── Save history ─────────────────────────────────────────────────────────
    hist_serialisable = {k: [float(v) for v in vals]
                         for k, vals in history.history.items()}
    with open(HISTORY_PATH, "w") as f:
        json.dump(hist_serialisable, f, indent=2)

    # ── Plot ─────────────────────────────────────────────────────────────────
    _plot_history(history.history)

    return history.history


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _plot_history(h: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(h["accuracy"],     label="Train acc")
    axes[0].plot(h["val_accuracy"], label="Val acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Loss
    axes[1].plot(h["loss"],     label="Train loss")
    axes[1].plot(h["val_loss"], label="Val loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    plt.close()
    print(f"[train] Training curves saved → {PLOT_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Model loader (shared by predict.py and app)
# ──────────────────────────────────────────────────────────────────────────────

def load_trained_model(path: str = MODEL_PATH) -> keras.Model:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}.\n"
            "Run training first:  python src/train.py"
        )
    print(f"[train] Loading model from {path} …")
    return keras.models.load_model(path)


def load_model_meta(path: str = META_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model metadata not found at {path}.")
    with open(path, "r") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else None
    train_model(dataset_root=root)
