"""Stage 5: tactical metrics in court coordinates (feet, with metric twins).

Everything here is pure pandas/numpy/scipy on projected tracks, so it is unit
tested on synthetic data. Frames whose calibration confidence is below
calibration.low_conf_threshold are excluded: metrics are NaN for those frames
instead of being computed from a bad homography.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError, Voronoi

from .court import FT_TO_M, CourtSpec

FTPS_TO_KMH = FT_TO_M * 3.6
FT2_TO_M2 = FT_TO_M**2
TEAMS = (0, 1)


# ------------------------------------------------------------------ geometry

def polygon_area(poly: np.ndarray) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_polygon_rect(poly: np.ndarray, rect) -> np.ndarray:
    """Sutherland-Hodgman clip of a polygon to an axis-aligned rectangle."""
    x0, y0, x1, y1 = rect
    out = np.asarray(poly, float)
    for axis, bound, keep_ge in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        if len(out) == 0:
            break
        inp, out_l = out, []
        for i in range(len(inp)):
            cur, prev = inp[i], inp[i - 1]
            cin = cur[axis] >= bound if keep_ge else cur[axis] <= bound
            pin = prev[axis] >= bound if keep_ge else prev[axis] <= bound
            if cin != pin:
                t = (bound - prev[axis]) / (cur[axis] - prev[axis])
                out_l.append(prev + t * (cur - prev))
            if cin:
                out_l.append(cur)
        out = np.array(out_l)
    return out


def hull_area(pts: np.ndarray) -> float:
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return np.nan
    try:
        return float(ConvexHull(pts).volume)  # 2-D "volume" is the area
    except QhullError:
        return 0.0  # collinear


def voronoi_cells(pts: np.ndarray, rect) -> list[np.ndarray]:
    """Voronoi cells of pts clipped to rect (exact, via mirroring the points
    across the four rectangle edges so every original cell is bounded)."""
    pts = np.asarray(pts, float)
    n = len(pts)
    if n == 0:
        return []
    if n == 1:
        x0, y0, x1, y1 = rect
        return [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)]
    x0, y0, x1, y1 = rect
    jitter = np.random.default_rng(0).normal(0, 1e-6, pts.shape)  # break exact duplicates
    p = np.clip(pts, [x0, y0], [x1, y1]) + jitter
    mirrored = [p,
                np.c_[2 * x0 - p[:, 0], p[:, 1]], np.c_[2 * x1 - p[:, 0], p[:, 1]],
                np.c_[p[:, 0], 2 * y0 - p[:, 1]], np.c_[p[:, 0], 2 * y1 - p[:, 1]]]
    vor = Voronoi(np.concatenate(mirrored))
    cells = []
    for i in range(n):
        reg = vor.regions[vor.point_region[i]]
        poly = vor.vertices[[v for v in reg if v >= 0]]
        c = poly.mean(0)
        order = np.argsort(np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0]))
        cells.append(clip_polygon_rect(poly[order], rect))  # clip removes float slop
    return cells


def space_control(pts: np.ndarray, teams: np.ndarray, rect, region=None) -> dict[int, float]:
    """Area (ft^2) controlled by each team. Cells are computed over the whole
    court `rect`; with `region` (e.g. the frontcourt) they are then clipped to it,
    so a player outside the region still owns the part of their cell inside it."""
    cells = voronoi_cells(pts, rect)
    out = {t: 0.0 for t in TEAMS}
    for c, t in zip(cells, teams):
        if t in out:
            out[t] += polygon_area(clip_polygon_rect(c, region) if region is not None else c)
    return out


def nearest_gap(att: np.ndarray, dfn: np.ndarray) -> np.ndarray:
    """Distance from each attacker to the nearest defender (ft)."""
    if len(att) == 0 or len(dfn) == 0:
        return np.full(len(att), np.nan)
    d = np.hypot(att[:, None, 0] - dfn[None, :, 0], att[:, None, 1] - dfn[None, :, 1])
    return d.min(1)


# ---------------------------------------------------------------- kinematics

def _fill_short_gaps(s: pd.Series, max_gap: int) -> pd.Series:
    """Linearly interpolate NaN runs of length <= max_gap; longer runs stay NaN
    (pandas' interpolate(limit=) would partially fill long gaps)."""
    isna = s.isna().to_numpy()
    filled = s.interpolate(limit_area="inside").to_numpy()
    out = s.to_numpy().copy()
    i, n = 0, len(out)
    while i < n:
        if isna[i]:
            j = i
            while j + 1 < n and isna[j + 1]:
                j += 1
            if j - i + 1 <= max_gap:
                out[i : j + 1] = filled[i : j + 1]
            i = j + 1
        else:
            i += 1
    return pd.Series(out, index=s.index)


def _smooth_runs(x: np.ndarray, w: int) -> np.ndarray:
    """Savitzky-Golay (quadratic) per contiguous finite run: smooths jitter but,
    unlike a moving average, does not shrink motion at the ends of a run."""
    from scipy.signal import savgol_filter

    out = np.full_like(x, np.nan)
    fin = np.isfinite(x)
    i, n = 0, len(x)
    while i < n:
        if not fin[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and fin[j + 1]:
            j += 1
        seg = x[i : j + 1]
        if len(seg) >= w:
            out[i : j + 1] = savgol_filter(seg, w, 2, mode="interp")
        elif len(seg) >= 3:
            out[i : j + 1] = savgol_filter(seg, len(seg) | 1 if len(seg) % 2 else len(seg) - 1, 1, mode="interp")
        else:
            out[i : j + 1] = seg
        i = j + 1
    return out


def track_kinematics(t: pd.DataFrame, fps: float, window_s: float, max_kmh: float, max_gap: int) -> pd.DataFrame:
    """For one track (rows sorted by frame, columns frame/x_ft/y_ft/valid):
    bridge short gaps, smooth positions over ~window_s, compute speed and step
    distance, and reject physically impossible steps (projection glitches, ID
    swaps) instead of letting them inflate distance."""
    f0, f1 = int(t.frame.min()), int(t.frame.max())
    s = pd.DataFrame(index=pd.RangeIndex(f0, f1 + 1, name="frame"))
    s = s.join(t[t.valid].set_index("frame")[["x_ft", "y_ft"]])
    x = _fill_short_gaps(s.x_ft, max_gap).to_numpy().copy()
    y = _fill_short_gaps(s.y_ft, max_gap).to_numpy().copy()
    w = max(5, int(round(window_s * fps)) | 1)
    # reject single-frame glitches before smoothing: a point far from the
    # median of its neighbourhood is a projection error, not motion
    med_x = pd.Series(x).rolling(5, center=True, min_periods=3).median().to_numpy()
    med_y = pd.Series(y).rolling(5, center=True, min_periods=3).median().to_numpy()
    max_step_ft = max_kmh / FTPS_TO_KMH / fps
    glitch = np.hypot(x - med_x, y - med_y) > 3 * max_step_ft
    x[glitch] = np.nan
    y[glitch] = np.nan
    x = _fill_short_gaps(pd.Series(x), max_gap).to_numpy()
    y = _fill_short_gaps(pd.Series(y), max_gap).to_numpy()
    xs, ys = _smooth_runs(x, w), _smooth_runs(y, w)
    step = np.hypot(np.diff(xs, prepend=np.nan), np.diff(ys, prepend=np.nan))
    spd = step * fps * FTPS_TO_KMH
    bad = spd > max_kmh
    step[bad] = np.nan
    spd[bad] = np.nan
    out = pd.DataFrame({"frame": np.arange(f0, f1 + 1), "x_s": xs, "y_s": ys,
                        "speed_kmh": pd.Series(spd).rolling(w, center=True, min_periods=2).mean().to_numpy(),
                        "step_ft": step})
    out["dist_m"] = (out.step_ft.fillna(0) * FT_TO_M).cumsum()
    return out


def add_kinematics(tracks: pd.DataFrame, fps: float, cfg) -> pd.DataFrame:
    m = cfg.metrics
    players = tracks[tracks.team.isin(TEAMS) & tracks.on_court].copy()
    players["valid"] = players.calib_ok & players.on_court_frame
    parts = []
    for tid, g in players.groupby("track_id"):
        k = track_kinematics(g.sort_values("frame"), fps, m.speed_window_s, m.max_speed_kmh, m.max_gap_frames)
        k["track_id"] = tid
        parts.append(k)
    if not parts:
        return tracks.assign(speed_kmh=np.nan, dist_m=np.nan, x_s=np.nan, y_s=np.nan, step_ft=np.nan)
    kin = pd.concat(parts)
    return tracks.merge(kin, on=["track_id", "frame"], how="left")


# --------------------------------------------------------------- per frame

def attacking_side(xs: np.ndarray, court: CourtSpec) -> str:
    return "right" if np.nanmean(xs) >= court.length / 2 else "left"


def offense_team(pos: dict[int, np.ndarray], side: str, court: CourtSpec) -> int | None:
    """In a half-court set the defence sits between the attackers and the basket:
    the team whose players are on average farther from the attacked hoop attacks."""
    hoop = court.hoop(side)
    d = {t: np.hypot(*(p - hoop).T).mean() for t, p in pos.items() if len(p) >= 2}
    if len(d) < 2:
        return None
    return max(d, key=d.get)


def frame_metrics(tracks: pd.DataFrame, cfg, start: int, end: int, ball: pd.DataFrame | None = None) -> pd.DataFrame:
    court = CourtSpec.from_cfg(cfg.court)
    full = (0.0, 0.0, court.length, court.width)
    pl = tracks[tracks.team.isin(TEAMS) & tracks.on_court & tracks.calib_ok & tracks.on_court_frame]
    pl = pl[np.isfinite(pl.x_s) & np.isfinite(pl.y_s)]
    by_f = {f: g for f, g in pl.groupby("frame")}
    ball_xy = {}
    if ball is not None and len(ball):
        ball_xy = {int(f): (x, y) for f, x, y in ball[["frame", "x_ft", "y_ft"]].itertuples(index=False)}
    rows = []
    for f in range(start, end):
        g = by_f.get(f)
        row = {"frame": f, "valid": False}
        if g is not None and len(g) >= 4 and g.team.nunique() == 2:
            pts = g[["x_s", "y_s"]].to_numpy()
            tm = g.team.to_numpy()
            pos = {t: pts[tm == t] for t in TEAMS}
            side = attacking_side(pts[:, 0], court)
            off = offense_team(pos, side, court)
            ctrl = space_control(pts, tm, full)
            front = space_control(pts, tm, full, court.half_rect(side))
            half_area = court.area / 2
            row.update(valid=True, attacking_side=side, offense=off if off is not None else -1, n_players=len(g))
            for t in TEAMS:
                row[f"hull_ft2_{t}"] = hull_area(pos[t]) if len(pos[t]) >= cfg.metrics.min_players_for_hull else np.nan
                row[f"control_pct_{t}"] = 100 * ctrl[t] / court.area
                row[f"front_control_pct_{t}"] = 100 * front[t] / half_area
            if off is not None:
                gaps = nearest_gap(pos[off], pos[1 - off])
                row["open_space_ft"] = float(np.nanmean(gaps)) if len(gaps) else np.nan
                # ball handler: offensive player nearest a detected (projected) ball
                b = ball_xy.get(f)
                if b is not None and len(pos[off]):
                    d = np.hypot(pos[off][:, 0] - b[0], pos[off][:, 1] - b[1])
                    i = int(np.argmin(d))
                    if d[i] <= cfg.possession.ball_assign_dist_ft:
                        row["handler_gap_ft"] = float(gaps[i])
        rows.append(row)
    fm = pd.DataFrame(rows)
    for col in ["hull_ft2_0", "hull_ft2_1", "control_pct_0", "control_pct_1", "front_control_pct_0",
                "front_control_pct_1", "open_space_ft", "handler_gap_ft", "offense", "attacking_side", "n_players"]:
        if col not in fm:
            fm[col] = np.nan
    return fm


def team_distance(tracks: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Cumulative distance (m) per team per frame, summed over that team's players."""
    st = tracks[tracks.team.isin(TEAMS)].groupby(["frame", "team"]).step_ft.sum(min_count=1).unstack()
    st = st.reindex(index=range(start, end), columns=list(TEAMS)).fillna(0) * FT_TO_M
    cum = st.cumsum()
    return pd.DataFrame({"frame": cum.index, "dist_m_0": cum[0].to_numpy(), "dist_m_1": cum[1].to_numpy(),
                         "step_m_0": st[0].to_numpy(), "step_m_1": st[1].to_numpy()})


# --------------------------------------------------------------- possessions

def segment_possessions(fm: pd.DataFrame, fps: float, smoothing_s: float, min_duration_s: float,
                        max_invalid_gap_s: float = 2.0) -> np.ndarray:
    """Possession id per frame (-1 = none). The per-frame label (offense team,
    attacking side) is majority-smoothed; a possession changes when the label
    changes (team-side change) or after a long run of invalid frames. Runs shorter
    than min_duration_s are absorbed into the neighbouring possession."""
    lab = np.where(fm.valid & (fm.offense >= 0),
                   fm.offense.fillna(-1).astype(int) * 2 + (fm.attacking_side == "right").astype(int), -1)
    w = max(1, int(round(smoothing_s * fps)))
    s = pd.Series(np.where(lab >= 0, lab, np.nan))
    sm = s.rolling(w, center=True, min_periods=1).apply(
        lambda a: pd.Series(a).mode().iloc[0] if np.isfinite(a).any() else np.nan, raw=True)
    lab_s = np.where(np.isfinite(sm), sm, -1).astype(int)
    lab_s[lab < 0] = -1
    # runs
    runs = []
    i = 0
    n = len(lab_s)
    while i < n:
        j = i
        while j + 1 < n and lab_s[j + 1] == lab_s[i]:
            j += 1
        runs.append([i, j, lab_s[i]])
        i = j + 1
    min_len = int(round(min_duration_s * fps))
    max_gap = int(round(max_invalid_gap_s * fps))
    # short labelled runs inherit the previous labelled run's label
    for r in runs:
        if r[2] >= 0 and r[1] - r[0] + 1 < min_len:
            prev = next((q[2] for q in reversed(runs[: runs.index(r)]) if q[2] >= 0), None)
            if prev is not None:
                r[2] = prev
    pid = np.full(n, -1)
    cur, last_lab, last_end = -1, None, -10**9
    for a, b, l in runs:
        if l < 0:
            continue
        if l != last_lab or a - last_end > max_gap:
            cur += 1
        pid[a : b + 1] = cur
        last_lab, last_end = l, b
    return pid


def possession_summary(fm: pd.DataFrame, td: pd.DataFrame, fps: float, labels, start: int) -> list[dict]:
    out = []
    fm = fm.merge(td, on="frame", how="left")
    for pid, lab in fm[fm.possession >= 0].groupby("possession"):
        f0, f1 = int(lab.frame.min()), int(lab.frame.max())
        g = fm[(fm.frame >= f0) & (fm.frame <= f1)]  # whole span, incl. low-confidence frames
        v = lab[lab.valid]
        off = int(v.offense.mode().iloc[0]) if len(v) else -1
        dfn = 1 - off if off in TEAMS else -1
        side = v.attacking_side.mode().iloc[0] if len(v) else None

        def avg(col):
            x = v[col].astype(float)
            return None if x.isna().all() else round(float(x.mean()), 2)

        rec = {
            "possession_id": int(pid),
            "start_frame": f0, "end_frame": f1,
            "start_s": round(f0 / fps, 2), "end_s": round((f1 + 1) / fps, 2),
            "duration_s": round((f1 - f0 + 1) / fps, 2),
            "offense": labels[off] if off in TEAMS else None,
            "defense": labels[dfn] if dfn in TEAMS else None,
            "attacking_side": side,
            "valid_frame_pct": round(100 * len(v) / len(g), 1),  # metrics below average these frames only
        }
        if off in TEAMS:
            for role, t in (("offense", off), ("defense", dfn)):
                a = avg(f"hull_ft2_{t}")
                rec[f"avg_spacing_{role}_ft2"] = a
                rec[f"avg_spacing_{role}_m2"] = None if a is None else round(a * FT2_TO_M2, 2)
                rec[f"avg_space_control_{role}_pct"] = avg(f"control_pct_{t}")
                rec[f"avg_frontcourt_control_{role}_pct"] = avg(f"front_control_pct_{t}")
                rec[f"distance_covered_{role}_m"] = round(float(g[f"step_m_{t}"].sum()), 1)
            o = avg("open_space_ft")
            rec["avg_open_space_ft"] = o
            rec["avg_open_space_m"] = None if o is None else round(o * FT_TO_M, 2)
            h = avg("handler_gap_ft")
            rec["avg_ball_handler_gap_ft"] = h
        out.append(rec)
    return out


# ------------------------------------------------------------------- driver

def project_ball(dets: pd.DataFrame, H: np.ndarray, conf: np.ndarray, start: int, thr: float) -> pd.DataFrame:
    """Best ball detection per frame, projected via its box bottom. The floor
    homography is only valid for points on the floor, so this is an approximation
    used solely to pick the ball handler (dribbles keep the ball low)."""
    from .court import apply_h

    b = dets[dets.cls == "ball"].sort_values("conf").groupby("frame").tail(1)
    rows = []
    for r in b.itertuples():
        k = r.frame - start
        if 0 <= k < len(conf) and conf[k] >= thr and np.isfinite(H[k]).all():
            xy = apply_h(np.linalg.inv(H[k]), np.array([[(r.x1 + r.x2) / 2, r.y2]]))[0]
            rows.append((r.frame, xy[0], xy[1]))
    return pd.DataFrame(rows, columns=["frame", "x_ft", "y_ft"])


def run_metrics(ptracks: pd.DataFrame, dets: pd.DataFrame, cal: dict, cfg, fps: float, start: int, end: int):
    labels = list(cfg.teams.labels)
    kin = add_kinematics(ptracks, fps, cfg)
    ball = project_ball(dets, cal["H"], cal["conf"], start, cfg.calibration.low_conf_threshold)
    fm = frame_metrics(kin, cfg, start, end, ball)
    td = team_distance(kin, start, end)
    fm["possession"] = segment_possessions(fm, fps, cfg.possession.smoothing_s, cfg.possession.min_duration_s)
    summary = possession_summary(fm, td, fps, labels, start)
    fm = fm.merge(td, on="frame", how="left")
    fm["calib_conf"] = cal["conf"][fm.frame.to_numpy() - start]

    role = np.where(kin.team.isin(TEAMS), "player", np.where(kin.team == 2, "referee", "unknown"))
    out = pd.DataFrame({
        "frame": kin.frame, "time_s": (kin.frame / fps).round(3), "segment": kin.segment,
        "track_id": kin.track_id, "team": kin.team,
        "team_label": kin.team.map({0: labels[0], 1: labels[1], 2: "REF/OTHER"}).fillna("UNKNOWN"),
        "role": role, "on_court": kin.on_court,
        "x_ft": kin.x_s.where(kin.x_s.notna(), kin.x_ft), "y_ft": kin.y_s.where(kin.y_s.notna(), kin.y_ft),
        "speed_kmh": kin.speed_kmh.round(2), "dist_covered_m": kin.dist_m.round(2),
        "foot_u": kin.foot_u.round(1), "foot_v": kin.foot_v.round(1),
        "x1": kin.x1, "y1": kin.y1, "x2": kin.x2, "y2": kin.y2, "det_id": kin.det_id,
        "calib_conf": kin.calib_conf.round(3), "valid": kin.calib_ok & kin.on_court_frame,
    })
    out["x_m"] = out.x_ft * FT_TO_M
    out["y_m"] = out.y_ft * FT_TO_M
    # positions from low-confidence frames are not trustworthy: blank them
    out.loc[~out.valid, ["x_ft", "y_ft", "x_m", "y_m", "speed_kmh"]] = np.nan
    return out.sort_values(["frame", "track_id"]).reset_index(drop=True), fm, summary
