"""Image -> court projection of player foot points, plus the on-court filter."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .court import FT_TO_M, CourtSpec, apply_h
from .detect import kpts_from_row

L_ANKLE, R_ANKLE = 15, 16


def foot_point(row, kp: np.ndarray | None, min_conf: float = 0.4) -> tuple[float, float]:
    """Ground contact point in image px. Ankles (pose) when both are confident:
    better than the box bottom for jumping or lunging players; else box bottom-centre."""
    if kp is not None:
        a = kp[[L_ANKLE, R_ANKLE]]
        if (a[:, 2] >= min_conf).all():
            x = float(a[:, 0].mean())
            y = float(max(a[:, 1].max(), row.y2 - 0.04 * (row.y2 - row.y1)))
            return x, y
    return (row.x1 + row.x2) / 2.0, float(row.y2)


def project_tracks(tracks: pd.DataFrame, dets: pd.DataFrame, H: np.ndarray, conf: np.ndarray,
                   start: int, cfg) -> pd.DataFrame:
    court = CourtSpec.from_cfg(cfg.court)
    thr = cfg.calibration.low_conf_threshold
    margin = cfg.court.on_court_margin_ft
    kp = dets[dets.cls == "person"].set_index(["frame", "det_id"]).kpts
    rows = []
    for r in tracks.itertuples():
        k = kpts_from_row(kp.get((r.frame, r.det_id)))
        u, v = foot_point(r, k)
        rows.append((u, v))
    uv = np.array(rows, float).reshape(-1, 2)
    out = tracks.copy()
    out["foot_u"], out["foot_v"] = uv[:, 0], uv[:, 1]
    idx = out.frame.to_numpy() - start
    c = conf[idx]
    xy = np.full((len(out), 2), np.nan)
    for f in np.unique(idx):
        Hf = H[f]
        if not np.isfinite(Hf).all():
            continue
        m = idx == f
        xy[m] = apply_h(np.linalg.inv(Hf), uv[m])
    out["x_ft"], out["y_ft"] = xy[:, 0], xy[:, 1]
    out["calib_conf"] = c
    out["calib_ok"] = (c >= thr) & np.isfinite(xy).all(1)
    inside = court.contains(np.nan_to_num(xy, nan=-1e9), margin)
    # per-frame decision where calibrated; per-track majority otherwise (bench/crowd
    # stays off-court even through low-confidence stretches)
    out["on_court_frame"] = out.calib_ok & inside
    frac = out[out.calib_ok].groupby("track_id").on_court_frame.mean()
    out["on_court"] = out.track_id.map(frac).fillna(0.0).to_numpy() >= 0.5
    out["x_m"], out["y_m"] = out.x_ft * FT_TO_M, out.y_ft * FT_TO_M
    return out
