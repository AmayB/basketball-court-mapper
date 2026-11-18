"""Scene-cut detection from two signals between consecutive frames:

1. structure: zero-normalized correlation of gradient-magnitude images computed on
   histogram-equalized grayscale. Robust to exposure/colour-grading flashes
   (which some broadcasts and replays apply) because equalization removes global
   brightness/contrast changes while the court lines and crowd stay in place.
2. colour: Bhattacharyya distance of HSV histograms (the classic cut signal).

A cut fires when structure collapses (corr < struct_threshold), or when the colour
histogram jumps and structure is at least moderately different. A colour jump
with intact structure is an exposure flash, not a cut.

Broadcasts cut between cameras, to replays and to close-ups. Tracking IDs and the
camera homography are only valid within one continuous shot, so the pipeline
splits the clip into segments at detected cuts and resets state at each boundary.
"""
from __future__ import annotations

import cv2
import numpy as np

from .video import iter_frames


def hist_feature(frame: np.ndarray, bins=(30, 32)) -> np.ndarray:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256])
    return cv2.normalize(h, None, alpha=1.0, norm_type=cv2.NORM_L1).astype(np.float32)


def struct_feature(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(cv2.equalizeHist(g).astype(np.float32), (3, 3), 0)
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return (m - m.mean()) / (m.std() + 1e-6)


def struct_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).mean())


def is_cut(hist_d: float, corr: float, sc) -> bool:
    return corr < sc.struct_threshold or (hist_d >= sc.threshold and corr < sc.hist_struct_threshold)


def hist_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


def detect_cuts(distances: np.ndarray, threshold: float, min_gap: int, flags=None) -> list[int]:
    """distances[i] = cut strength between frame i-1 and i (index 0 ignored).
    Candidates are flags[i] (if given) else distances[i] >= threshold. Candidates
    closer than min_gap are merged keeping the strongest (dissolves, flashes)."""
    d = np.asarray(distances, float)
    if flags is None:
        flags = d >= threshold
    cand = [i for i in range(1, len(d)) if flags[i]]
    cuts: list[int] = []
    for i in cand:
        if cuts and i - cuts[-1] < min_gap:
            if d[i] > d[cuts[-1]]:
                cuts[-1] = i
            continue
        cuts.append(i)
    return cuts


def segments_from_cuts(start: int, end: int, cuts: list[int]) -> list[tuple[int, int]]:
    """Half-open [a, b) frame ranges between cuts (absolute frame indices)."""
    bounds = [start] + [c for c in cuts if start < c < end] + [end]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def run_scenecut(video: str, cfg, start: int, end: int) -> dict:
    sc = cfg.scenecut
    hd, corr = [], []
    prev = None
    for _, frame in iter_frames(video, start, end - start):
        cur = (hist_feature(frame, sc.hist_bins), struct_feature(frame))
        hd.append(0.0 if prev is None else hist_distance(prev[0], cur[0]))
        corr.append(1.0 if prev is None else struct_corr(prev[1], cur[1]))
        prev = cur
    hd_a, corr_a = np.array(hd), np.array(corr)
    flags = np.array([is_cut(h, c, sc) for h, c in zip(hd_a, corr_a)])
    strength = 1.0 - corr_a
    cuts_rel = detect_cuts(strength, 0.0, sc.min_gap_frames, flags)
    cuts = [start + c for c in cuts_rel]
    return {"cuts": cuts, "hist_distance": hd_a.round(4).tolist(), "struct_corr": corr_a.round(4).tolist(),
            "segments": segments_from_cuts(start, end, cuts)}
