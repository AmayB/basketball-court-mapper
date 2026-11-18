"""Stage 4: team assignment from jersey colour.

Per detection: crop the torso (upper-middle of the box), convert to Lab, mask
pixels close to this frame's floor colour (the floor shows through between arms
and torso), and summarise the rest as [L - L_floor, a, b, std(L)]. Expressing
lightness relative to the floor makes the feature tolerant to exposure changes
(flashes, replays), and std(L) separates striped referee shirts from solid
jerseys. KMeans (k=3) clusters all samples; each track then takes the majority
vote of its samples, so a track's label is stable over time.

Cluster roles: the cluster with the lowest average number of simultaneous
on-court people per frame is "referee/other" (3 refs vs 5 per team; track
counts are a worse signal because player tracks fragment more under occlusion).
Of the remaining two, the lighter
jersey is HOME when teams.home_jersey == "light" (the usual home convention),
otherwise the darker one. Display labels come from config (default HOME/AWAY).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .video import iter_frames

REF = 2  # team index used for referees / other


def torso_crop(frame: np.ndarray, box, tx=(0.28, 0.72), ty=(0.18, 0.50)) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    xa, xb = int(x1 + tx[0] * w), int(x1 + tx[1] * w)
    ya, yb = int(y1 + ty[0] * h), int(y1 + ty[1] * h)
    H, W = frame.shape[:2]
    xa, xb, ya, yb = max(0, xa), min(W, xb), max(0, ya), min(H, yb)
    if xb - xa < 2 or yb - ya < 2:
        return np.zeros((0, 0, 3), np.uint8)
    return frame[ya:yb, xa:xb]


def jersey_feature(crop_bgr: np.ndarray, floor_lab: np.ndarray, court_dist: float, min_pixels: int):
    """[L - L_floor, a, b, std(L)] of non-floor torso pixels, or None."""
    if crop_bgr.size == 0:
        return None
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    keep = np.linalg.norm(lab - floor_lab, axis=1) > court_dist
    px = lab[keep]
    if len(px) < min_pixels:
        return None
    med = np.median(px, 0)
    return np.array([med[0] - floor_lab[0], med[1], med[2], px[:, 0].std()], np.float32)


def frame_floor_lab(frame: np.ndarray) -> np.ndarray:
    H, W = frame.shape[:2]
    band = frame[int(H * 0.6) : int(H * 0.9), int(W * 0.1) : int(W * 0.9)]
    lab = cv2.cvtColor(band, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return np.median(lab, 0)


def cluster_features(F: np.ndarray, k: int = 3, seed: int = 0):
    """Standardise features (std(L) down-weighted) and run KMeans."""
    mu, sd = F.mean(0), F.std(0) + 1e-6
    w = np.array([1.0, 1.0, 1.0, 0.6], np.float32)
    Z = (F - mu) / sd * w
    km = KMeans(k, n_init=10, random_state=seed).fit(Z)
    centers = km.cluster_centers_ / w * sd + mu
    return km.labels_, centers


def vote(track_ids: np.ndarray, labels: np.ndarray, min_votes: int) -> dict[int, tuple[int, float, int]]:
    """Majority cluster per track: {track_id: (cluster, purity, n_votes)}."""
    out = {}
    for tid in np.unique(track_ids):
        lab = labels[track_ids == tid]
        if len(lab) < min_votes:
            continue
        counts = np.bincount(lab, minlength=3)
        k = int(np.argmax(counts))
        out[int(tid)] = (k, float(counts[k] / len(lab)), int(len(lab)))
    return out


def assign_roles(centers: np.ndarray, presence: np.ndarray, home_jersey: str = "light") -> dict[int, int]:
    """Map cluster -> team index (0 = HOME, 1 = AWAY, 2 = referee/other).
    presence[k] = mean number of cluster-k people visible per frame."""
    ref = int(np.argmin(presence))
    teams = [k for k in range(len(centers)) if k != ref]
    light, dark = sorted(teams, key=lambda k: -centers[k][0])
    home, away = (light, dark) if home_jersey == "light" else (dark, light)
    return {home: 0, away: 1, ref: REF}


def final_labels(track_ids: np.ndarray, labels: np.ndarray, role: dict[int, int], min_votes: int,
                 ref_min_share: float = 0.6) -> dict[int, int]:
    """Per-track team. A track is referee/other only when that cluster clearly
    dominates its votes; otherwise it takes the majority of the two team
    clusters. Occluded or motion-blurred crops tend to fall into the catch-all
    cluster, so a plain majority would lose real players to it."""
    team = np.array([role[int(k)] for k in labels])
    out = {}
    for tid in np.unique(track_ids):
        tm = team[track_ids == tid]
        if len(tm) < min_votes:
            continue
        c = np.bincount(tm, minlength=3)
        if c[REF] / len(tm) >= ref_min_share or c[0] + c[1] == 0:
            out[int(tid)] = REF
        else:
            out[int(tid)] = int(np.argmax(c[:2]))
    return out


def cluster_presence(frames: np.ndarray, track_ids: np.ndarray, track_cluster: dict, k: int = 3) -> np.ndarray:
    """Mean per-frame count of people in each cluster (over frames with anyone)."""
    cl = np.array([track_cluster.get(int(t), -1) for t in track_ids])
    df = pd.DataFrame({"frame": frames, "cl": cl})
    df = df[df.cl >= 0]
    if df.empty:
        return np.zeros(k)
    counts = df.groupby(["frame", "cl"]).size().unstack(fill_value=0).reindex(columns=range(k), fill_value=0)
    return counts.mean(0).to_numpy()


def run_teams(video: str, tracks: pd.DataFrame, cfg, start: int, end: int, overlay: np.ndarray | None = None) -> dict:
    """tracks must have an 'on_court' column (from projection). Returns
    {track_team: {tid: team}, purity: {...}, centers: [...]}."""
    t = cfg.teams
    on = tracks[tracks.on_court]
    by_frame = {f: g for f, g in on.groupby("frame")}
    feats, tids = [], []
    for fi, frame in iter_frames(video, start, end - start):
        if (fi - start) % max(1, int(t.sample_every)) or fi not in by_frame:
            continue
        floor = frame_floor_lab(frame)
        for r in by_frame[fi].itertuples():
            if overlay is not None and overlay[int(r.y1):int(r.y2), int(r.x1):int(r.x2)].mean() > 0.2:
                continue  # torso hidden behind a broadcast graphic
            f = jersey_feature(torso_crop(frame, (r.x1, r.y1, r.x2, r.y2), t.torso_x, t.torso_y),
                               floor, t.court_color_dist, t.min_pixels)
            if f is not None:
                feats.append(f)
                tids.append(r.track_id)
    if len(feats) < 10:
        return {"track_team": {}, "purity": {}, "centers": []}
    F = np.stack(feats)
    tids_a = np.array(tids)
    labels, centers = cluster_features(F)
    votes = vote(tids_a, labels, int(t.min_votes))
    presence = cluster_presence(on.frame.to_numpy(), on.track_id.to_numpy(), {tid: v[0] for tid, v in votes.items()})
    role = assign_roles(centers, presence, t.home_jersey)
    team_of = final_labels(tids_a, labels, role, int(t.min_votes), float(t.ref_min_share))
    return {
        "track_team": team_of,
        "purity": {tid: p for tid, (_, p, _) in votes.items()},
        "centers": {role[k]: centers[k].round(2).tolist() for k in range(len(centers))},
        "n_samples": int(len(F)),
        "presence_per_frame": {role[k]: round(float(presence[k]), 2) for k in range(len(centers))},
    }
