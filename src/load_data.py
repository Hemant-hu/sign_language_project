"""
load_data.py
============
Step 1 — Load the WLASL-processed dataset from KaggleHub.

Real dataset structure (what KaggleHub actually delivers):
──────────────────────────────────────────────────────────
  <root>/
    WLASL_v0.3.json        ← full annotations: [{gloss, instances:[{video_id,...}]}]
    nslt_100.json          ← 100-class split:  {train:[ids], val:[ids], test:[ids]}
    nslt_300.json
    nslt_1000.json
    nslt_2000.json
    videos/
      00000.mp4
      00001.mp4
      …

Pipeline:
  1. Read nslt_100.json  → get all video_ids
  2. Read WLASL_v0.3.json → build {video_id: gloss} mapping
  3. For each video: OpenCV frames → MediaPipe hand landmarks → (SEQ_LEN, 63) npy
  4. Cache results so MediaPipe only runs once
  5. Return X (N, SEQ_LEN, 63)  and  y (N,) label strings

Run standalone:
    python src/load_data.py [dataset_root] [max_videos]
"""

from __future__ import annotations

import os
import sys
import json
import collections
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import Optional

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SEQ_LEN       = 20    # frames per sample fed to the LSTM
NUM_LANDMARKS = 21    # MediaPipe Hands keypoints
NUM_FEATURES  = NUM_LANDMARKS * 3   # x, y, z → 63

# Which nslt subset to use: 100 / 300 / 1000 / 2000
# 100 → fastest; increase for a richer vocabulary
NSLT_CLASSES  = 100

_ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR  = os.path.join(_ROOT_DIR, "data", "cache")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Download
# ──────────────────────────────────────────────────────────────────────────────

def download_dataset() -> str:
    try:
        import kagglehub
    except ImportError:
        raise ImportError("Run:  pip install kagglehub")
    print("[load_data] Downloading WLASL dataset from KaggleHub …")
    path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")
    print(f"[load_data] Dataset root: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# 2. Exploration report
# ──────────────────────────────────────────────────────────────────────────────

def explore_dataset(root: str) -> None:
    print("\n" + "=" * 60)
    print("  DATASET EXPLORATION REPORT")
    print("=" * 60)
    print(f"  Root : {root}")

    ext_counts: collections.Counter = collections.Counter()
    for _, _, files in os.walk(root):
        for f in files:
            ext_counts[Path(f).suffix.lower()] += 1

    print("\n  File types found:")
    for ext, cnt in sorted(ext_counts.items(), key=lambda x: -x[1]):
        print(f"    {ext or '(none)':>8}  →  {cnt:,}")

    top = sorted(os.listdir(root))
    print(f"\n  Top-level entries ({len(top)}):")
    for e in top[:25]:
        kind = "DIR " if os.path.isdir(os.path.join(root, e)) else "FILE"
        print(f"    [{kind}] {e}")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 3. JSON parsing — handles the real WLASL annotation format
# ──────────────────────────────────────────────────────────────────────────────

def _build_videoid_to_gloss(annotation_path: str) -> dict[str, str]:
    """
    Parse WLASL_v0.3.json.

    Actual format:
        [
          {
            "gloss": "book",
            "instances": [
              {"video_id": "00000", "split": "train", ...},
              ...
            ]
          },
          ...
        ]

    Returns {video_id_str: GLOSS_LABEL} — labels uppercased, spaces→underscore.
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Normalise to a flat list of entries
    if isinstance(raw, dict):
        # Rare wrapper format — convert values to list
        entries = list(raw.values())
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(f"Unexpected top-level JSON type: {type(raw)}")

    mapping: dict[str, str] = {}
    skipped_entries = 0

    for entry in entries:
        # Guard: entry must be a dict with a 'gloss' key
        if not isinstance(entry, dict):
            skipped_entries += 1
            continue

        gloss = entry.get("gloss", "UNKNOWN")
        if not isinstance(gloss, str):
            skipped_entries += 1
            continue

        label = gloss.strip().replace(" ", "_").upper()

        instances = entry.get("instances", [])
        if not isinstance(instances, list):
            continue

        for inst in instances:
            if not isinstance(inst, dict):
                continue
            vid_id = inst.get("video_id", "")
            # Normalise to 5-digit zero-padded string
            try:
                vid_id_str = str(int(vid_id)).zfill(5)
            except (ValueError, TypeError):
                vid_id_str = str(vid_id).zfill(5)
            mapping[vid_id_str] = label

    if skipped_entries:
        print(f"[load_data] Skipped {skipped_entries} malformed annotation entries.")
    print(f"[load_data] Annotation map built: {len(mapping):,} video_id → gloss pairs")
    return mapping


def _load_nslt_split(nslt_path: str) -> dict[str, list[str]]:
    """
    Parse nslt_100.json (or nslt_300/1000/2000).

    Format:
        {"train": ["00000", "00001", ...], "val": [...], "test": [...]}

    Returns the dict with video_ids normalised to 5-digit strings.
    """
    with open(nslt_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    result: dict[str, list[str]] = {}
    for split, ids in raw.items():
        normalised = []
        for i in ids:
            try:
                normalised.append(str(int(i)).zfill(5))
            except (ValueError, TypeError):
                normalised.append(str(i).zfill(5))
        result[split] = normalised
        print(f"[load_data] Split '{split}': {len(normalised):,} videos")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 4. Video → landmark sequence
# ──────────────────────────────────────────────────────────────────────────────

def _extract_frames(video_path: str, n_frames: int = 60) -> list[np.ndarray]:
    """Sample n_frames evenly from the video; return list of BGR frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    n       = min(n_frames, total)
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def _normalize_landmarks(raw: np.ndarray) -> np.ndarray:
    """
    raw : (63,) — 21 landmarks × (x, y, z)
    Translate so wrist (landmark 0) is at origin, then scale to [-1, 1].
    """
    pts   = raw.reshape(21, 3).copy()
    pts  -= pts[0]
    scale = np.abs(pts).max()
    if scale > 1e-6:
        pts /= scale
    return pts.flatten().astype(np.float32)


def _video_to_sequence(video_path: str, hands) -> Optional[np.ndarray]:
    """
    Full pipeline for one video:
      frames → MediaPipe → normalise → pad/trim → (SEQ_LEN, 63)
    Returns None if fewer than 4 hand-detected frames.
    """
    frames = _extract_frames(video_path, n_frames=60)
    if not frames:
        return None

    landmark_frames: list[np.ndarray] = []
    for frame in frames:
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if not results.multi_hand_landmarks:
            continue
        lms = results.multi_hand_landmarks[0].landmark
        raw = np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten()
        landmark_frames.append(_normalize_landmarks(raw))

    if len(landmark_frames) < 4:
        return None

    L = len(landmark_frames)
    if L >= SEQ_LEN:
        idxs = np.linspace(0, L - 1, SEQ_LEN, dtype=int)
        seq  = np.array([landmark_frames[i] for i in idxs])
    else:
        pad = SEQ_LEN - L
        seq = np.array(landmark_frames + [landmark_frames[-1]] * pad)

    assert seq.shape == (SEQ_LEN, NUM_FEATURES), f"Bad shape: {seq.shape}"
    return seq.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Master loader
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(
    root: str | None = None,
    nslt_classes: int = NSLT_CLASSES,
    max_videos: int | None = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Download (if needed), process, and return (X, y).

    X : float32, shape (N, SEQ_LEN, NUM_FEATURES) = (N, 20, 63)
    y : str,     shape (N,)  — gloss label per sample

    Parameters
    ----------
    root          : local dataset path; None → download automatically
    nslt_classes  : vocabulary size (100 / 300 / 1000 / 2000)
    max_videos    : cap total videos for quick smoke-tests
    use_cache     : load/save processed arrays to data/cache/ for fast re-runs
    """
    if not _MP_AVAILABLE:
        raise ImportError(
            "mediapipe is required.\n"
            "Run:  pip install mediapipe"
        )

    if root is None:
        root = download_dataset()

    explore_dataset(root)

    # ── Locate required files ────────────────────────────────────────────────
    annotation_file = os.path.join(root, "WLASL_v0.3.json")
    nslt_file       = os.path.join(root, f"nslt_{nslt_classes}.json")
    videos_dir      = os.path.join(root, "videos")

    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"WLASL_v0.3.json not found under {root}")
    if not os.path.exists(nslt_file):
        fallback = os.path.join(root, "nslt_100.json")
        if os.path.exists(fallback):
            print(f"[load_data] nslt_{nslt_classes}.json not found; using nslt_100.json")
            nslt_file = fallback
        else:
            raise FileNotFoundError(f"nslt_{nslt_classes}.json not found under {root}")
    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(f"'videos/' folder not found under {root}")

    # ── Cache ────────────────────────────────────────────────────────────────
    tag     = f"wlasl{nslt_classes}_seq{SEQ_LEN}_feat{NUM_FEATURES}"
    tag    += f"_max{max_videos}" if max_videos else ""
    cache_X = os.path.join(CACHE_DIR, f"{tag}_X.npy")
    cache_y = os.path.join(CACHE_DIR, f"{tag}_y.npy")

    if use_cache and os.path.exists(cache_X) and os.path.exists(cache_y):
        print(f"[load_data] ✓ Loading from cache ({CACHE_DIR}) …")
        X = np.load(cache_X).astype(np.float32)
        y = np.load(cache_y, allow_pickle=True)
        _print_summary(X, y)
        return X, y

    # ── Parse JSON ───────────────────────────────────────────────────────────
    vid2gloss = _build_videoid_to_gloss(annotation_file)
    splits    = _load_nslt_split(nslt_file)

    # Collect all unique video_ids across all splits
    seen: set[str] = set()
    all_ids: list[str] = []
    for split_ids in splits.values():
        for vid in split_ids:
            if vid not in seen:
                seen.add(vid)
                all_ids.append(vid)

    if max_videos:
        import random; random.shuffle(all_ids)
        all_ids = all_ids[:max_videos]
        print(f"[load_data] Capped at {max_videos} videos (smoke-test mode).")

    print(f"\n[load_data] Processing {len(all_ids):,} videos with MediaPipe …")
    print(f"[load_data] First run takes a few minutes. Results are cached afterwards.\n")

    # ── MediaPipe processing ──────────────────────────────────────────────────
    mp_hands  = mp.solutions.hands
    X_list: list[np.ndarray] = []
    y_list: list[str]        = []
    skipped = 0

    hands_cfg = dict(
        static_image_mode        = False,
        max_num_hands            = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )

    with mp_hands.Hands(**hands_cfg) as hands:
        for vid_id in tqdm(all_ids, desc="Landmarks", unit="vid"):
            # Try zero-padded filename first, then un-padded
            vpath = os.path.join(videos_dir, f"{vid_id}.mp4")
            if not os.path.exists(vpath):
                vpath = os.path.join(videos_dir, f"{int(vid_id)}.mp4")
            if not os.path.exists(vpath):
                skipped += 1
                continue

            label = vid2gloss.get(vid_id)
            if label is None:
                skipped += 1
                continue

            seq = _video_to_sequence(vpath, hands)
            if seq is None:
                skipped += 1
                continue

            X_list.append(seq)
            y_list.append(label)

    print(f"\n[load_data] Extracted : {len(X_list):,} sequences")
    print(f"[load_data] Skipped   : {skipped:,} (missing file / no hand detected)")

    if not X_list:
        raise RuntimeError(
            "No sequences were extracted.\n"
            "Check that:\n"
            "  • videos/*.mp4 files actually exist\n"
            "  • mediapipe is installed correctly\n"
            "  • video_ids in JSON match filenames in videos/"
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)

    # ── Save cache ────────────────────────────────────────────────────────────
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(cache_X, X)
        np.save(cache_y, y)
        print(f"[load_data] Cached X → {cache_X}")
        print(f"[load_data] Cached y → {cache_y}")

    _print_summary(X, y)
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# 6. Summary printer
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(X: np.ndarray, y: np.ndarray) -> None:
    unique, counts = np.unique(y, return_counts=True)
    print("\n" + "─" * 55)
    print("  DATASET SUMMARY")
    print("─" * 55)
    print(f"  Total samples   : {len(X):,}")
    print(f"  Unique classes  : {len(unique)}")
    print(f"  Sequence shape  : {X.shape[1:]}")
    print(f"  X dtype         : {X.dtype}")
    print(f"\n  Top 15 classes by count:")
    top = np.argsort(-counts)[:15]
    for i in top:
        bar = "█" * min(int(counts[i] / counts.max() * 25), 25)
        print(f"    {unique[i]:>25}  {counts[i]:>4}  {bar}")
    print("─" * 55 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root_arg    = sys.argv[1] if len(sys.argv) > 1 else None
    max_vid_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    X, y = load_dataset(root_arg, max_videos=max_vid_arg)
    print(f"X : {X.shape}  dtype={X.dtype}")
    print(f"y : {y.shape}  sample={y[:5]}")
