"""
load_data.py
============
Step 1 — Load the WLASL-processed dataset from KaggleHub.

Actual dataset structure
────────────────────────
  <root>/
    WLASL_v0.3.json        ← [{gloss, instances:[{video_id,...}]}]
    nslt_100.json          ← {video_id: "train"|"val"|"test"}   ← flat id→split map
    nslt_300.json
    nslt_1000.json
    nslt_2000.json
    wlasl_class_list.txt
    videos/
      00000.mp4  …  11979.mp4

Pipeline
────────
  1. Read WLASL_v0.3.json  →  {video_id: gloss} map
  2. Read nslt_100.json    →  invert {video_id: split} to get all valid ids
  3. For each video: OpenCV frames → MediaPipe HandLandmarker → (SEQ_LEN, 63) array
  4. Cache results to data/cache/ so re-runs are instant
  5. Return X (N, 20, 63), y (N,) string labels

MediaPipe version
─────────────────
  mediapipe >= 0.10 removed mp.solutions.hands.
  We use the new Tasks API: mp.tasks.vision.HandLandmarker.
  The model file (hand_landmarker.task) is auto-downloaded on first run.

Run standalone:
    python src/load_data.py [dataset_root] [max_videos]
"""

from __future__ import annotations

import os
import sys
import json
import collections
import urllib.request
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import Optional

import mediapipe as mp

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SEQ_LEN       = 20   # frames per sample
NUM_LANDMARKS = 21   # MediaPipe hand landmarks
NUM_FEATURES  = NUM_LANDMARKS * 3   # x, y, z → 63
NSLT_CLASSES  = 100  # default subset (100 / 300 / 1000 / 2000)

_ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR  = os.path.join(_ROOT_DIR, "data", "cache")
MODEL_DIR  = os.path.join(_ROOT_DIR, "data", "mediapipe_models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. MediaPipe model download
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_mp_model() -> str:
    """
    Download the MediaPipe HandLandmarker model file if not already cached.
    Returns the local path to the .task file.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    print(f"[load_data] Downloading MediaPipe hand landmarker model …")
    print(f"  URL : {MODEL_URL}")
    print(f"  To  : {MODEL_PATH}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(f"\r  Progress: {pct:5.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=_progress)
        print()  # newline after progress
        print(f"[load_data] Model saved → {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download MediaPipe model: {e}\n"
            f"Download manually from:\n  {MODEL_URL}\n"
            f"Save to: {MODEL_PATH}"
        )
    return MODEL_PATH


# ──────────────────────────────────────────────────────────────────────────────
# 2. KaggleHub download
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
# 3. Exploration report
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
# 4. JSON parsing
# ──────────────────────────────────────────────────────────────────────────────

def _build_videoid_to_gloss(annotation_path: str) -> dict[str, str]:
    """
    Parse WLASL_v0.3.json.

    Format:
        [{"gloss": "book", "instances": [{"video_id": "00000", ...}]}, ...]

    Returns {video_id_zfill5: GLOSS_LABEL}.
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entries = raw if isinstance(raw, list) else list(raw.values())

    mapping: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        gloss = entry.get("gloss", "UNKNOWN")
        if not isinstance(gloss, str):
            continue
        label = gloss.strip().replace(" ", "_").upper()
        for inst in entry.get("instances", []):
            if not isinstance(inst, dict):
                continue
            try:
                vid_id = str(int(inst.get("video_id", ""))).zfill(5)
            except (ValueError, TypeError):
                continue
            mapping[vid_id] = label

    print(f"[load_data] Annotation map: {len(mapping):,} video_id → gloss entries")
    return mapping


def _load_nslt_ids(nslt_path: str) -> list[str]:
    """
    Parse nslt_XXX.json.

    The ACTUAL format in this Kaggle dataset is:
        {"28208": "train", "28205": "val", "00001": "test", ...}
        i.e.  {video_id: split_name}  — keys ARE the video ids.

    (Not {split_name: [video_ids]} as one might expect.)

    Returns a deduplicated list of all video_id strings (zero-padded to 5 digits).
    """
    with open(nslt_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict in {nslt_path}, got {type(raw)}")

    # Peek at the structure to auto-detect format
    first_key   = next(iter(raw))
    first_value = raw[first_key]

    all_ids: list[str] = []

    if isinstance(first_value, str):
        # Format A:  {video_id: split_name}
        # e.g. {"05237": "train", "05238": "test", ...}
        split_counts: collections.Counter = collections.Counter()
        for vid_id, split in raw.items():
            try:
                norm = str(int(vid_id)).zfill(5)
            except (ValueError, TypeError):
                norm = str(vid_id).zfill(5)
            all_ids.append(norm)
            split_counts[str(split)] += 1
        print(f"[load_data] nslt split distribution: {dict(split_counts)}")

    elif isinstance(first_value, list):
        # Format B:  {split_name: [video_ids]}
        # e.g. {"train": ["05237", ...], "test": [...]}
        for split, ids in raw.items():
            for vid in ids:
                try:
                    norm = str(int(vid)).zfill(5)
                except (ValueError, TypeError):
                    norm = str(vid).zfill(5)
                all_ids.append(norm)
            print(f"[load_data] Split '{split}': {len(ids):,} videos")

    elif isinstance(first_value, dict):
        # Format C (actual Kaggle version):  {video_id: {metadata_dict}}
        # e.g. {"05237": {"split": "train", "signer_id": 1, ...}, ...}
        # Keys ARE the video_ids; extract split info for logging if present.
        split_counts: collections.Counter = collections.Counter()
        for vid_id, meta in raw.items():
            try:
                norm = str(int(vid_id)).zfill(5)
            except (ValueError, TypeError):
                norm = str(vid_id).zfill(5)
            all_ids.append(norm)
            # Log split distribution if metadata contains a split field
            split_val = meta.get("split") or meta.get("subset") or "unknown"
            split_counts[str(split_val)] += 1
        print(f"[load_data] nslt split distribution: {dict(split_counts)}")

    else:
        raise ValueError(
            f"Unrecognised nslt JSON format.\n"
            f"First key: {first_key!r}, first value type: {type(first_value)}\n"
            f"Please open {nslt_path} and check its structure."
        )

    # Deduplicate, preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for vid in all_ids:
        if vid not in seen:
            seen.add(vid)
            unique.append(vid)

    print(f"[load_data] Total unique video IDs in nslt file: {len(unique):,}")
    return unique


# ──────────────────────────────────────────────────────────────────────────────
# 5. Frame extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_frames(video_path: str, n_frames: int = 60) -> list[np.ndarray]:
    """Uniformly sample n_frames BGR frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    n       = min(n_frames, total)
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames: list[np.ndarray] = []
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
    Translate so wrist (idx 0) is at origin, scale to [-1, 1].
    """
    pts   = raw.reshape(21, 3).copy()
    pts  -= pts[0]
    scale = np.abs(pts).max()
    if scale > 1e-6:
        pts /= scale
    return pts.flatten().astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 6. MediaPipe HandLandmarker (new Tasks API, mediapipe >= 0.10)
# ──────────────────────────────────────────────────────────────────────────────

def _make_landmarker(model_path: str):
    """
    Build a MediaPipe HandLandmarker using the new Tasks API (mp >= 0.10).
    Returns the landmarker context manager.
    """
    VisionTask          = mp.tasks.vision
    BaseOptions         = mp.tasks.BaseOptions
    HandLandmarker      = VisionTask.HandLandmarker
    HandLandmarkerOptions = VisionTask.HandLandmarkerOptions
    RunningMode         = VisionTask.RunningMode

    options = HandLandmarkerOptions(
        base_options                  = BaseOptions(model_asset_path=model_path),
        running_mode                  = RunningMode.IMAGE,
        num_hands                     = 1,
        min_hand_detection_confidence = 0.5,
        min_hand_presence_confidence  = 0.5,
        min_tracking_confidence       = 0.5,
    )
    return HandLandmarker.create_from_options(options)


def _detect_landmarks(frame_bgr: np.ndarray, landmarker) -> Optional[np.ndarray]:
    """
    Run HandLandmarker on one BGR frame.
    Returns normalised (63,) array, or None if no hand detected.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result    = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    # result.hand_landmarks[0] is a list of 21 NormalizedLandmark objects
    lms = result.hand_landmarks[0]
    raw = np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten()
    return _normalize_landmarks(raw)


def _video_to_sequence(
    video_path: str,
    landmarker,
) -> Optional[np.ndarray]:
    """
    Full single-video pipeline:
        frames → landmarks → normalise → pad/trim → (SEQ_LEN, 63)
    Returns None if fewer than 4 hand-detected frames.
    """
    frames = _extract_frames(video_path, n_frames=60)
    if not frames:
        return None

    detected: list[np.ndarray] = []
    for frame in frames:
        lm = _detect_landmarks(frame, landmarker)
        if lm is not None:
            detected.append(lm)

    if len(detected) < 4:
        return None

    L = len(detected)
    if L >= SEQ_LEN:
        idxs = np.linspace(0, L - 1, SEQ_LEN, dtype=int)
        seq  = np.array([detected[i] for i in idxs])
    else:
        pad = SEQ_LEN - L
        seq = np.array(detected + [detected[-1]] * pad)

    return seq.astype(np.float32)   # (SEQ_LEN, 63)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Master loader
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
    root          : local dataset path; None → auto-download via KaggleHub
    nslt_classes  : vocabulary size — 100 / 300 / 1000 / 2000
    max_videos    : cap total videos (useful for quick smoke-tests, e.g. 50)
    use_cache     : load/save processed arrays for fast re-runs
    """
    if root is None:
        root = download_dataset()

    explore_dataset(root)

    # ── Locate required files ────────────────────────────────────────────────
    annotation_file = os.path.join(root, "WLASL_v0.3.json")
    videos_dir      = os.path.join(root, "videos")

    # Find the best available nslt file
    nslt_file = None
    for n in [nslt_classes, 100, 300, 1000, 2000]:
        candidate = os.path.join(root, f"nslt_{n}.json")
        if os.path.exists(candidate):
            nslt_file = candidate
            if n != nslt_classes:
                print(f"[load_data] nslt_{nslt_classes}.json not found; "
                      f"using nslt_{n}.json instead")
            break

    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"WLASL_v0.3.json not found in {root}")
    if nslt_file is None:
        raise FileNotFoundError(f"No nslt_XXX.json found in {root}")
    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(f"'videos/' folder not found in {root}")

    # ── Cache check ──────────────────────────────────────────────────────────
    tag     = f"wlasl{nslt_classes}_seq{SEQ_LEN}"
    tag    += f"_max{max_videos}" if max_videos else ""
    cache_X = os.path.join(CACHE_DIR, f"{tag}_X.npy")
    cache_y = os.path.join(CACHE_DIR, f"{tag}_y.npy")

    if use_cache and os.path.exists(cache_X) and os.path.exists(cache_y):
        print(f"[load_data] ✓ Cache hit — loading from {CACHE_DIR}")
        X = np.load(cache_X).astype(np.float32)
        y = np.load(cache_y, allow_pickle=True)
        _print_summary(X, y)
        return X, y

    # ── Parse JSON files ─────────────────────────────────────────────────────
    vid2gloss = _build_videoid_to_gloss(annotation_file)
    all_ids   = _load_nslt_ids(nslt_file)

    if max_videos:
        import random
        random.shuffle(all_ids)
        all_ids = all_ids[:max_videos]
        print(f"[load_data] Capped at {max_videos} videos (smoke-test mode).")

    print(f"\n[load_data] Processing {len(all_ids):,} videos with MediaPipe …")
    print(f"[load_data] First run takes several minutes; results cached afterwards.\n")

    # ── Ensure MediaPipe model ────────────────────────────────────────────────
    mp_model_path = _ensure_mp_model()

    # ── Processing loop ───────────────────────────────────────────────────────
    X_list: list[np.ndarray] = []
    y_list: list[str]        = []
    skipped = 0

    with _make_landmarker(mp_model_path) as landmarker:
        for vid_id in tqdm(all_ids, desc="Extracting landmarks", unit="vid"):
            # Try zero-padded filename, then integer filename
            vpath = os.path.join(videos_dir, f"{vid_id}.mp4")
            if not os.path.exists(vpath):
                try:
                    vpath = os.path.join(videos_dir, f"{int(vid_id)}.mp4")
                except ValueError:
                    pass
            if not os.path.exists(vpath):
                skipped += 1
                continue

            label = vid2gloss.get(vid_id)
            if label is None:
                skipped += 1
                continue

            seq = _video_to_sequence(vpath, landmarker)
            if seq is None:
                skipped += 1
                continue

            X_list.append(seq)
            y_list.append(label)

    print(f"\n[load_data] Extracted : {len(X_list):,} sequences")
    print(f"[load_data] Skipped   : {skipped:,}  "
          f"(missing file / no hand detected / no gloss mapping)")

    if not X_list:
        raise RuntimeError(
            "No sequences could be extracted.\n"
            "Common causes:\n"
            "  • videos/*.mp4 files are missing or corrupted\n"
            "  • MediaPipe model file failed to download\n"
            "  • video_ids in JSON don't match filenames in videos/\n"
            "Try running with max_videos=10 to debug a small subset."
        )

    X = np.array(X_list, dtype=np.float32)   # (N, 20, 63)
    y = np.array(y_list)                       # (N,)

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
# 8. Summary printer
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(X: np.ndarray, y: np.ndarray) -> None:
    unique, counts = np.unique(y, return_counts=True)
    print("\n" + "─" * 55)
    print("  DATASET SUMMARY")
    print("─" * 55)
    print(f"  Total samples   : {len(X):,}")
    print(f"  Unique classes  : {len(unique)}")
    print(f"  Sequence shape  : {X.shape[1:]}   (frames × features)")
    print(f"  X dtype         : {X.dtype}")
    print(f"\n  Top 15 classes by sample count:")
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
    print(f"\nX : {X.shape}  dtype={X.dtype}")
    print(f"y : {y.shape}  sample={list(y[:8])}")
