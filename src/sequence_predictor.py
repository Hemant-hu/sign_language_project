"""
sequence_predictor.py
=====================
Step 6 — Next-sign prediction using a bigram (conditional frequency) model
built from the training labels.

The bigram model stores:
    P(next_sign | current_sign)  =  count(current → next) / count(current)

It is trained automatically from the label array during the training phase
and persisted as  models/bigram_model.json.

Usage
-----
from src.sequence_predictor import BigramPredictor

bp = BigramPredictor.load()          # load from disk
bp.predict_next("WANT")              # → "FOOD"
bp.predict_next_n("I", n=3)          # → ["WANT", "LIKE", "NEED"]
"""

from __future__ import annotations

import os
import json
import collections
import numpy as np
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIGRAM_PATH = os.path.join(ROOT_DIR, "models", "bigram_model.json")


# ──────────────────────────────────────────────────────────────────────────────
# Bigram model
# ──────────────────────────────────────────────────────────────────────────────

class BigramPredictor:
    """
    Conditional frequency (bigram) model over sign sequences.

    Attributes
    ----------
    bigrams : dict[str, dict[str, int]]
        {sign_A: {sign_B: count, …}, …}
    """

    # ── Common sense priors ──────────────────────────────────────────────────
    # These cover cold-start (sign never seen in training transitions).
    _DEFAULT_TRANSITIONS: dict[str, list[str]] = {
        "I":        ["WANT", "LIKE", "NEED", "SEE", "GO", "EAT"],
        "YOU":      ["WANT", "GO", "EAT", "LIKE", "SEE"],
        "WE":       ["GO", "EAT", "NEED", "WANT", "MAKE"],
        "WANT":     ["FOOD", "WATER", "GO", "EAT", "DRINK"],
        "NEED":     ["FOOD", "WATER", "HELP", "GO", "WORK"],
        "LIKE":     ["FOOD", "SCHOOL", "WORK", "EAT"],
        "EAT":      ["FOOD", "WHAT", "WHERE", "YES", "NO"],
        "DRINK":    ["WATER", "WHAT", "YES", "NO"],
        "GO":       ["HOME", "SCHOOL", "WORK", "WHERE"],
        "HELLO":    ["HOW", "WHAT", "YOU", "I"],
        "THANK_YOU":["YES", "OK", "BYE"],
        "SORRY":    ["PLEASE", "NO", "YES"],
        "WHAT":     ["YOU", "WE", "I", "WANT", "NEED"],
        "WHERE":    ["HOME", "SCHOOL", "WORK", "GO"],
        "HOW":      ["YOU", "I", "WE", "GO"],
        "PLEASE":   ["HELP", "GO", "GIVE", "MAKE"],
        "GIVE":     ["FOOD", "WATER", "PLEASE"],
        "SEE":      ["YOU", "I", "HOME", "SCHOOL"],
        "MAKE":     ["FOOD", "WORK", "HOME"],
    }

    def __init__(self) -> None:
        # bigrams[A][B] = number of times B followed A in training data
        self.bigrams: dict[str, dict[str, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
        self._inject_priors()

    # ── Training ─────────────────────────────────────────────────────────────

    def _inject_priors(self) -> None:
        for sign, nexts in self._DEFAULT_TRANSITIONS.items():
            for nxt in nexts:
                self.bigrams[sign][nxt] += 1

    def fit(self, label_sequences: list[list[str]]) -> "BigramPredictor":
        """
        Learn transitions from sequences of sign labels.

        Parameters
        ----------
        label_sequences : list of lists
            Each inner list is an ordered sequence of sign labels
            (e.g. [["I", "WANT", "FOOD"], ["HELLO", "HOW"]])
        """
        n_pairs = 0
        for seq in label_sequences:
            for a, b in zip(seq[:-1], seq[1:]):
                self.bigrams[a][b] += 1
                n_pairs += 1
        print(f"[bigram] Trained on {n_pairs:,} bigram pairs from "
              f"{len(label_sequences):,} sequences.")
        return self

    def fit_from_flat_labels(self, labels: np.ndarray,
                             window: int = 3) -> "BigramPredictor":
        """
        Convenience method: given the flat label array from the dataset,
        create pseudo-sequences using a sliding window and fit.

        This approximates real sentence order when actual sentence
        grouping metadata is unavailable.
        """
        seqs: list[list[str]] = []
        lbl_list = labels.tolist()
        for i in range(0, len(lbl_list) - window + 1, 1):
            seqs.append(lbl_list[i: i + window])
        return self.fit(seqs)

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict_next(self, current_sign: str) -> Optional[str]:
        """
        Return the single most likely sign to follow current_sign,
        or None if current_sign has never been seen.
        """
        results = self.predict_next_n(current_sign, n=1)
        return results[0] if results else None

    def predict_next_n(self, current_sign: str, n: int = 3) -> list[str]:
        """
        Return the top-n most likely next signs.

        Falls back to global frequency ranking if current_sign is unknown.
        """
        candidates = self.bigrams.get(current_sign)
        if not candidates:
            # global fallback: most frequent sign overall
            global_counts: dict[str, int] = collections.Counter()
            for sub in self.bigrams.values():
                for sign, cnt in sub.items():
                    global_counts[sign] += cnt
            return [s for s, _ in global_counts.most_common(n)]

        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
        return [sign for sign, _ in sorted_candidates[:n]]

    def predict_from_history(
        self, history: list[str], n: int = 1
    ) -> list[str]:
        """
        Given a full history of recently predicted signs, use the last sign
        to predict the next.  Designed for the real-time prediction pipeline.

        Parameters
        ----------
        history : list[str]
            Ordered list of recently predicted signs, e.g. ["I", "WANT"]
        n : int
            Number of suggestions to return.
        """
        if not history:
            return []
        return self.predict_next_n(history[-1], n=n)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = BIGRAM_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # defaultdict is not JSON-serialisable directly
        serialisable = {k: dict(v) for k, v in self.bigrams.items()}
        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"[bigram] Model saved → {path}")

    @classmethod
    def load(cls, path: str = BIGRAM_PATH) -> "BigramPredictor":
        obj = cls()
        if not os.path.exists(path):
            print(f"[bigram] No saved model at {path}. Using default priors.")
            return obj
        with open(path, "r") as f:
            data = json.load(f)
        for sign, nexts in data.items():
            for nxt, cnt in nexts.items():
                obj.bigrams[sign][nxt] = cnt
        print(f"[bigram] Model loaded from {path}  "
              f"({len(obj.bigrams)} known signs)")
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Sentence formatter
# ──────────────────────────────────────────────────────────────────────────────

def signs_to_sentence(signs: list[str]) -> str:
    """
    Convert a list of sign labels to a readable English sentence.

    Rules:
      • THANK_YOU  →  "thank you"
      • Underscores replaced with spaces
      • First letter capitalised
    """
    words = [s.lower().replace("_", " ") for s in signs]
    sentence = " ".join(words).strip()
    return sentence.capitalize() if sentence else ""


# ──────────────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bp = BigramPredictor.load()

    tests = ["I", "HELLO", "WANT", "GO", "EAT", "THANK_YOU", "UNKNOWN_SIGN"]
    print("\nBigram predictions:")
    print(f"  {'Sign':<15}  {'Next (top 3)'}")
    print(f"  {'─'*15}  {'─'*30}")
    for sign in tests:
        nexts = bp.predict_next_n(sign, n=3)
        print(f"  {sign:<15}  {nexts}")

    demo = ["I", "WANT", "FOOD"]
    print(f"\nSentence from {demo}:")
    print(f"  → \"{signs_to_sentence(demo)}\"")
