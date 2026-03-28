"""
load_data.py
============
Step 1 — Download the WLASL-processed dataset from KaggleHub,
automatically explore its structure, and return clean (X, y) arrays.

Supported dataset layouts (auto-detected):
  A) Flat .npy files:  X.npy / y.npy  or  features.npy / labels.npy
  B) Per-class folders of .npy sequences  (e.g.  data/HELLO/0.npy …)
  C) A single CSV/JSON manifest with a 'path' + 'label' column
  D) Pre-split train/test sub-folders

Run standalone for a quick dataset report:
    python src/load_data.py
"""

from __future__ import annotations

import os
import json
import glob
import pickle
import collections
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Download
# ──────────────────────────────────────────────────────────────────────────────

def download_dataset() -> str:
    """
    Download the WLASL-processed dataset via KaggleHub.
    Returns the local root path (str).
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub is not installed.\n"
            "Run:  pip install kagglehub"
        )

    print("[load_data] Downloading dataset from KaggleHub …")
    path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")
    print(f"[load_data] Dataset available at: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Exploration helpers
# ──────────────────────────────────────────────────────────────────────────────

def _walk_extensions(root: str) -> collections.Counter:
    """Return a Counter of file extensions found under root."""
    counter: collections.Counter = collections.Counter()
    for _, _, files in os.walk(root):
        for f in files:
            ext = Path(f).suffix.lower()
            counter[ext] += 1
    return counter


def explore_dataset(root: str) -> None:
    """
    Print a human-readable summary of the dataset directory.
    Called automatically during loading but can be run standalone.
    """
    print("\n" + "=" * 60)
    print("  DATASET EXPLORATION REPORT")
    print("=" * 60)
    print(f"  Root : {root}")

    ext_counts = _walk_extensions(root)
    print(f"\n  File types found:")
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        print(f"    {ext or '(no ext)':>10}  →  {count:,} files")

    # Top-level structure
    top = sorted(os.listdir(root))
    print(f"\n  Top-level entries ({len(top)} total):")
    for entry in top[:20]:
        full = os.path.join(root, entry)
        kind = "DIR " if os.path.isdir(full) else "FILE"
        print(f"    [{kind}] {entry}")
    if len(top) > 20:
        print(f"    … and {len(top) - 20} more")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Layout detectors
# ──────────────────────────────────────────────────────────────────────────────

def _find_file(root: str, *names: str) -> str | None:
    """Return the first existing file matching any of names (case-insensitive)."""
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower() in [n.lower() for n in names]:
                return os.path.join(dirpath, f)
    return None


def _try_layout_flat_npy(root: str) -> tuple | None:
    """Layout A: X.npy + y.npy (or features.npy + labels.npy) in the same dir."""
    x_file = _find_file(root, "X.npy", "features.npy", "x.npy", "data.npy")
    y_file = _find_file(root, "y.npy", "labels.npy", "targets.npy", "label.npy")
    if x_file and y_file:
        print(f"[load_data] Layout A detected: flat .npy files")
        print(f"  X → {x_file}")
        print(f"  y → {y_file}")
        X = np.load(x_file, allow_pickle=True)
        y = np.load(y_file, allow_pickle=True)
        return X, y
    return None


def _try_layout_class_folders(root: str) -> tuple | None:
    """
    Layout B: root/CLASS_NAME/*.npy
    Each subfolder is a class; each .npy file is one sequence sample.
    """
    subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subdirs:
        return None

    # Check that at least some subdirs contain .npy files
    npy_subdirs = [
        d for d in subdirs
        if glob.glob(os.path.join(root, d, "*.npy"))
    ]
    if not npy_subdirs:
        return None

    print(f"[load_data] Layout B detected: per-class folders "
          f"({len(npy_subdirs)} classes with .npy files)")

    X_list, y_list = [], []
    for class_name in sorted(npy_subdirs):
        class_dir = os.path.join(root, class_name)
        npy_files = sorted(glob.glob(os.path.join(class_dir, "*.npy")))
        for fpath in npy_files:
            seq = np.load(fpath, allow_pickle=True)
            X_list.append(seq)
            y_list.append(class_name)

    if not X_list:
        return None

    # Handle jagged sequences: pad to max length
    X, y = _pad_sequences(X_list), np.array(y_list)
    return X, y


def _try_layout_csv_manifest(root: str) -> tuple | None:
    """Layout C: CSV manifest with columns [path, label] or [file, gloss]."""
    csv_file = _find_file(root, "wlasl_class_list.txt", "labels.csv",
                          "manifest.csv", "data.csv", "index.csv")
    if csv_file is None:
        # try any .csv
        csvs = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
        if csvs:
            csv_file = csvs[0]

    if csv_file is None:
        return None

    print(f"[load_data] Layout C detected: CSV manifest → {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  Columns: {list(df.columns)}")

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    path_col  = next((c for c in df.columns if c in ("path", "file", "filepath", "video")), None)
    label_col = next((c for c in df.columns if c in ("label", "gloss", "class", "sign", "target")), None)

    if path_col is None or label_col is None:
        print(f"  [WARN] Cannot map columns to (path, label). Skipping CSV layout.")
        return None

    X_list, y_list = [], []
    for _, row in df.iterrows():
        fpath = row[path_col]
        if not os.path.isabs(fpath):
            fpath = os.path.join(root, fpath)
        if not os.path.exists(fpath):
            continue
        seq = np.load(fpath, allow_pickle=True)
        X_list.append(seq)
        y_list.append(str(row[label_col]))

    if not X_list:
        return None
    return _pad_sequences(X_list), np.array(y_list)


def _try_layout_train_test_split(root: str) -> tuple | None:
    """Layout D: root/train/ and root/test/ sub-folders (each with class folders)."""
    train_dir = os.path.join(root, "train")
    if not os.path.isdir(train_dir):
        return None

    print(f"[load_data] Layout D detected: pre-split train/test folders")

    def load_split(split_dir: str):
        xs, ys = [], []
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fpath in sorted(glob.glob(os.path.join(class_dir, "*.npy"))):
                xs.append(np.load(fpath, allow_pickle=True))
                ys.append(class_name)
        return xs, ys

    X_train_l, y_train_l = load_split(train_dir)
    test_dir = os.path.join(root, "test")
    if os.path.isdir(test_dir):
        X_test_l, y_test_l = load_split(test_dir)
        all_X = X_train_l + X_test_l
        all_y = y_train_l + y_test_l
    else:
        all_X, all_y = X_train_l, y_train_l

    if not all_X:
        return None
    return _pad_sequences(all_X), np.array(all_y)


def _try_layout_wlasl_json(root: str) -> tuple | None:
    """
    Layout E: WLASL2000.json (official WLASL annotation format)
    + video feature .npy files stored alongside.
    """
    json_file = _find_file(root, "WLASL2000.json", "wlasl2000.json",
                           "wlasl_annotation.json", "nslt_2000.json")
    if json_file is None:
        return None

    print(f"[load_data] Layout E detected: WLASL JSON annotations → {json_file}")
    with open(json_file, "r") as f:
        data = json.load(f)

    X_list, y_list = [], []
    for entry in data:
        gloss = entry.get("gloss", "").replace(" ", "_").upper()
        for instance in entry.get("instances", []):
            vid_id = instance.get("video_id", "")
            # look for a matching .npy file anywhere under root
            candidates = glob.glob(os.path.join(root, "**", f"{vid_id}*.npy"),
                                   recursive=True)
            if not candidates:
                continue
            seq = np.load(candidates[0], allow_pickle=True)
            X_list.append(seq)
            y_list.append(gloss)

    if not X_list:
        return None
    return _pad_sequences(X_list), np.array(y_list)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Sequence padding utility
# ──────────────────────────────────────────────────────────────────────────────

def _pad_sequences(seqs: list, target_len: int | None = None) -> np.ndarray:
    """
    Pad / truncate a list of variable-length numpy arrays to the same length.
    Each array can be 1-D (features,) or 2-D (timesteps, features).
    Returns shape (N, target_len, features).
    """
    # Ensure each element is at least 2-D
    seqs_2d = []
    for s in seqs:
        s = np.array(s, dtype=np.float32)
        if s.ndim == 1:
            s = s.reshape(1, -1)   # treat whole array as one timestep
        seqs_2d.append(s)

    # Decide target length
    lengths = [s.shape[0] for s in seqs_2d]
    if target_len is None:
        target_len = int(np.median(lengths))   # use median to avoid outlier bias
        print(f"[load_data] Sequence lengths: min={min(lengths)}, "
              f"max={max(lengths)}, median={target_len} → using {target_len}")

    features = seqs_2d[0].shape[1]
    result   = np.zeros((len(seqs_2d), target_len, features), dtype=np.float32)

    for i, s in enumerate(seqs_2d):
        L = min(s.shape[0], target_len)
        result[i, :L, :] = s[:L]

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Master load function
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(root: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Auto-detect dataset layout and return (X, y).

    X : float32 array of shape  (N, sequence_length, features)
    y : str array of shape  (N,)  containing class names

    Parameters
    ----------
    root : str, optional
        Path returned by kagglehub.dataset_download().
        If None, downloads automatically.
    """
    if root is None:
        root = download_dataset()

    explore_dataset(root)

    # Try layouts in order of specificity
    for detector in [
        _try_layout_flat_npy,
        _try_layout_train_test_split,
        _try_layout_wlasl_json,
        _try_layout_class_folders,
        _try_layout_csv_manifest,
    ]:
        result = detector(root)
        if result is not None:
            X, y = result
            break
    else:
        raise RuntimeError(
            f"Could not auto-detect dataset layout under: {root}\n"
            "Supported layouts:\n"
            "  A) X.npy + y.npy\n"
            "  B) CLASS_NAME/*.npy folders\n"
            "  C) CSV manifest with [path, label] columns\n"
            "  D) train/ and test/ sub-folders\n"
            "  E) WLASL2000.json + per-video .npy files\n"
            "Please inspect the dataset and adapt load_data.py accordingly."
        )

    # ── Ensure 3-D ──────────────────────────────────────────────────────────
    if X.ndim == 2:
        # (N, features) → (N, 1, features)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        print(f"[load_data] Reshaped 2-D array → {X.shape}")

    # ── Final report ────────────────────────────────────────────────────────
    unique_labels, counts = np.unique(y, return_counts=True)
    print("\n" + "─" * 50)
    print("  DATASET SUMMARY")
    print("─" * 50)
    print(f"  Total samples     : {len(X):,}")
    print(f"  Unique classes    : {len(unique_labels)}")
    print(f"  Sequence shape    : {X.shape[1:]}")
    print(f"  X dtype           : {X.dtype}")
    print(f"\n  Class distribution (top 15):")
    top_idx = np.argsort(-counts)[:15]
    for i in top_idx:
        bar = "█" * min(int(counts[i] / counts.max() * 20), 20)
        print(f"    {unique_labels[i]:>20}  {counts[i]:>4}  {bar}")
    print("─" * 50 + "\n")

    return X.astype(np.float32), y


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    root_path = sys.argv[1] if len(sys.argv) > 1 else None
    X, y = load_dataset(root_path)
    print(f"Loaded  X={X.shape}  y={y.shape}")
