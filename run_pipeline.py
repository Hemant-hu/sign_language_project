"""
run_pipeline.py
===============
Master pipeline runner.  Run the entire project end-to-end:

    python run_pipeline.py [--dataset-root /path/to/data] [--epochs 30]

Steps
-----
1. Download + explore dataset
2. Preprocess (normalise, encode, split)
3. Train LSTM model
4. Build bigram next-sign model from training labels
5. Smoke-test prediction with a synthetic sequence
6. Print instructions to start the API
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import numpy as np

# ── ensure project root is on path ───────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sign Language to Text — full pipeline")
    p.add_argument("--dataset-root", default=None,
                   help="Local path returned by kagglehub (skip download if set)")
    p.add_argument("--epochs",       type=int, default=30,
                   help="Training epochs (default: 30)")
    p.add_argument("--batch-size",   type=int, default=32)
    p.add_argument("--skip-train",   action="store_true",
                   help="Skip training (use existing model)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────

def step_banner(n: int, title: str) -> None:
    line = "─" * 55
    print(f"\n{line}")
    print(f"  STEP {n}  |  {title}")
    print(f"{line}")


def main() -> None:
    args = parse_args()

    # ── Step 1: Data ─────────────────────────────────────────────────────────
    step_banner(1, "Data loading & exploration")
    from src.load_data import load_dataset
    X, y = load_dataset(args.dataset_root)

    # ── Step 2: Preprocess ───────────────────────────────────────────────────
    step_banner(2, "Preprocessing")
    from src.preprocess import (
        normalize_X, encode_labels, to_categorical,
        split_data, save_label_encoder
    )
    X_norm        = normalize_X(X)
    y_int, le     = encode_labels(y)
    num_classes   = len(le.classes_)
    y_onehot      = to_categorical(y_int, num_classes)
    X_train, X_test, y_train, y_test = split_data(X_norm, y_onehot)
    save_label_encoder(le)
    seq_len, n_feat = X_train.shape[1], X_train.shape[2]

    # ── Step 3 & 4: Training ─────────────────────────────────────────────────
    if not args.skip_train:
        step_banner(3, "Model training")
        from src.train import build_model, MODEL_PATH
        import tensorflow as tf
        from tensorflow.keras import callbacks as cb

        model = build_model(seq_len, n_feat, num_classes)
        model.summary()

        ckpt_cb = cb.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                                     save_best_only=True, verbose=1)
        es_cb   = cb.EarlyStopping(monitor="val_loss", patience=7,
                                   restore_best_weights=True, verbose=1)
        lr_cb   = cb.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                       patience=3, min_lr=1e-6, verbose=1)

        history = model.fit(
            X_train, y_train,
            epochs          = args.epochs,
            batch_size      = args.batch_size,
            validation_split= 0.15,
            callbacks       = [ckpt_cb, es_cb, lr_cb],
            verbose         = 1,
        )

        # Evaluate
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n  Test accuracy : {acc:.4f}  ({acc*100:.1f}%)")
        print(f"  Test loss     : {loss:.4f}")

        # Save metadata
        meta = dict(
            sequence_length = seq_len,
            num_features    = n_feat,
            num_classes     = num_classes,
            test_accuracy   = float(acc),
            test_loss       = float(loss),
            epochs_trained  = len(history.history["loss"]),
        )
        meta_path = os.path.join(ROOT, "models", "model_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Metadata → {meta_path}")

    else:
        print("[run] Skipping training — using existing model.")

    # ── Step 5: Bigram model from training labels ────────────────────────────
    step_banner(5, "Next-sign predictor (bigram)")
    from src.sequence_predictor import BigramPredictor
    bp = BigramPredictor()
    bp.fit_from_flat_labels(y, window=3)
    bp.save()

    # ── Step 6: Smoke test ───────────────────────────────────────────────────
    step_banner(6, "Smoke-test prediction")
    from src.predict import SignPredictor

    try:
        predictor = SignPredictor()
        test_seq  = np.random.rand(seq_len, n_feat).astype(np.float32)
        result    = predictor.predict(test_seq)

        print(f"\n  Input shape     : ({seq_len}, {n_feat})")
        print(f"  Current sign    : {result['current_sign']}")
        print(f"  Confidence      : {result['confidence']:.1%}")
        print(f"  Next prediction : {result['next_prediction']}")
        print(f"  Sentence        : \"{result['sentence']}\"")
        print(f"\n  Top predictions :")
        for p in result["top_predictions"]:
            bar = "█" * int(p["confidence"] * 20)
            print(f"    {p['sign']:>20}  {p['confidence']:.1%}  {bar}")
    except Exception as e:
        print(f"  [WARN] Smoke test failed: {e}")

    # ── Done ─────────────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  ✓  Pipeline complete!")
    print("═" * 55)
    print("\nTo start the API:")
    print("  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    print("\nAPI docs available at:")
    print("  http://localhost:8000/docs")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()
