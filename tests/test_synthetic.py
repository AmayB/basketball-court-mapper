"""Synthetic-data checks: fake players on a known court seen through a known camera.

1. homography math: project court points to the image and back; re-estimate H
   from noisy correspondences with RANSAC.
2. Voronoi space control: cells tile the court exactly (areas sum to 94x50 ft).
3. speed/distance: a player running at a known constant speed.
"""
import cv2
import numpy as np
import pandas as pd
import pytest

from hoopsvision.config import load_config
from hoopsvision.court import FT_TO_M, CourtSpec, apply_h
from hoopsvision.metrics import (
    clip_polygon_rect,
    frame_metrics,
    hull_area,
    nearest_gap,
    offense_team,
    polygon_area,
    segment_possessions,
    space_control,
    track_kinematics,
    voronoi_cells,
)

COURT = CourtSpec()
FPS = 30.0


def camera_h():
    src = np.array([[0, 0], [94, 0], [94, 50], [0, 50]], np.float32)
    dst = np.array([[-300, 700], [1580, 690], [1100, 180], [180, 185]], np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


# ---------------------------------------------------------------- homography

def test_homography_roundtrip_and_ransac_recovery():
    H = camera_h()
    rng = np.random.default_rng(0)
    players = rng.uniform([0, 0], [94, 50], (10, 2))
    uv = apply_h(H, players)
    back = apply_h(np.linalg.inv(H), uv)
    assert np.allclose(back, players, atol=1e-9)

    # landmarks with 0.5 px noise + 20% gross outliers -> RANSAC recovers H
    lm = COURT.sample_points(2.0)
    obs = apply_h(H, lm) + rng.normal(0, 0.5, (len(lm), 2))
    bad = rng.random(len(lm)) < 0.2
    obs[bad] += rng.uniform(-80, 80, (bad.sum(), 2))
    Hest, inl = cv2.findHomography(lm.astype(np.float32), obs.astype(np.float32), cv2.RANSAC, 2.0)
    err_ft = np.hypot(*(apply_h(np.linalg.inv(Hest), uv) - players).T)
    assert err_ft.max() < 0.25, err_ft  # well under a foot on the court


# ------------------------------------------------------------------- voronoi

@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_voronoi_cells_tile_the_court(n):
    pts = np.random.default_rng(n).uniform([0, 0], [94, 50], (n, 2))
    cells = voronoi_cells(pts, (0, 0, 94, 50))
    total = sum(polygon_area(c) for c in cells)
    assert total == pytest.approx(94 * 50, rel=1e-6)
    for p, c in zip(pts, cells):  # each player lies in their own cell
        assert cv2.pointPolygonTest(c.astype(np.float32).reshape(-1, 1, 2), tuple(map(float, p)), False) >= 0


def test_space_control_symmetry_and_frontcourt():
    # mirror-symmetric formation: each team controls exactly half
    a = np.array([[20, 10], [30, 25], [20, 40]], float)
    b = np.c_[94 - a[:, 0], a[:, 1]]
    pts = np.vstack([a, b])
    teams = np.array([0, 0, 0, 1, 1, 1])
    ctrl = space_control(pts, teams, (0, 0, 94, 50))
    assert ctrl[0] == pytest.approx(ctrl[1]) == pytest.approx(94 * 50 / 2)
    front = space_control(pts, teams, (0, 0, 94, 50), COURT.half_rect("right"))
    assert sum(front.values()) == pytest.approx(47 * 50, rel=1e-6)
    assert front[1] > front[0]


def test_clip_and_hull():
    sq = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]], float)
    assert polygon_area(clip_polygon_rect(sq, (0, 0, 94, 50))) == pytest.approx(100)
    assert hull_area(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]])) == pytest.approx(100)
    assert np.isnan(hull_area(np.array([[0, 0], [1, 1]])))
    assert nearest_gap(np.array([[0, 0.0]]), np.array([[3, 4.0], [10, 0]]))[0] == pytest.approx(5)


# --------------------------------------------------------------------- speed

def _run(speed_mps=5.0, seconds=4.0, spike_at=None):
    n = int(seconds * FPS)
    v_ft = speed_mps / FT_TO_M / FPS  # ft per frame
    x = 10 + v_ft * np.arange(n)
    y = np.full(n, 25.0)
    if spike_at is not None:
        x = x.copy()
        x[spike_at] += 15  # one-frame projection glitch (15 ft jump)
    return pd.DataFrame({"frame": np.arange(n), "x_ft": x, "y_ft": y, "valid": True})


def test_speed_and_distance_known_trajectory():
    k = track_kinematics(_run(5.0), FPS, 0.5, 36.0, 6)
    mid = k.speed_kmh.iloc[20:-20]
    assert mid.mean() == pytest.approx(18.0, rel=0.02)  # 5 m/s = 18 km/h
    total_true = 5.0 * (len(k) - 1) / FPS
    assert k.dist_m.iloc[-1] == pytest.approx(total_true, rel=0.05)


def test_speed_spike_is_rejected_not_counted():
    clean = track_kinematics(_run(5.0), FPS, 0.5, 36.0, 6)
    spiky = track_kinematics(_run(5.0, spike_at=60), FPS, 0.5, 36.0, 6)
    assert spiky.speed_kmh.max() <= 36.0
    # the glitch must not add ~15+ ft of phantom distance
    assert spiky.dist_m.iloc[-1] < clean.dist_m.iloc[-1] * 1.08


def test_gap_interpolation_limits():
    df = _run(4.0)
    df.loc[30:33, "valid"] = False   # 4-frame occlusion: bridged
    df.loc[60:80, "valid"] = False   # 21-frame gap: not bridged
    k = track_kinematics(df, FPS, 0.5, 36.0, 6)
    assert k.loc[k.frame.between(30, 33), "x_s"].notna().all()
    assert k.loc[k.frame.between(62, 78), "x_s"].isna().all()


# --------------------------------------------------------- frames/possession

def _synthetic_tracks(frames, offense_right_team=0):
    """5 attackers spread around the right arc, 5 defenders packed near the rim."""
    rows = []
    hoop = COURT.hoop("right")
    for f in frames:
        for i in range(5):
            ang = np.radians(-70 + 35 * i)
            att = hoop + 24 * np.array([-np.cos(ang), np.sin(ang)])
            dfn = hoop + 14 * np.array([-np.cos(ang), np.sin(ang)])
            for team, p, tid in ((offense_right_team, att, i), (1 - offense_right_team, dfn, 10 + i)):
                rows.append(dict(frame=f, track_id=tid, team=team, x_s=p[0], y_s=p[1], on_court=True,
                                 calib_ok=True, on_court_frame=True))
    return pd.DataFrame(rows)


def test_frame_metrics_offense_and_open_space():
    cfg = load_config(None)
    tr = _synthetic_tracks(range(5))
    fm = frame_metrics(tr, cfg, 0, 5)
    assert fm.valid.all()
    assert (fm.offense == 0).all() and (fm.attacking_side == "right").all()
    assert fm.open_space_ft.iloc[0] == pytest.approx(10.0, abs=1e-6)
    assert fm.hull_ft2_0.iloc[0] > fm.hull_ft2_1.iloc[0]          # offense spaced wider
    assert (fm.control_pct_0 + fm.control_pct_1).iloc[0] == pytest.approx(100.0)
    pos = {0: tr[(tr.frame == 0) & (tr.team == 0)][["x_s", "y_s"]].to_numpy(),
           1: tr[(tr.frame == 0) & (tr.team == 1)][["x_s", "y_s"]].to_numpy()}
    assert offense_team(pos, "right", COURT) == 0


def test_possession_segmentation_on_team_side_change():
    n = int(12 * FPS)
    fm = pd.DataFrame({"frame": np.arange(n), "valid": True,
                       "offense": np.where(np.arange(n) < n // 2, 0, 1),
                       "attacking_side": np.where(np.arange(n) < n // 2, "right", "left")})
    fm.loc[100:103, "offense"] = 1  # 4-frame flicker must not create a possession
    pid = segment_possessions(fm, FPS, 1.0, 2.5)
    assert set(np.unique(pid)) == {0, 1}
    assert pid[n // 2 - 5] == 0 and pid[n // 2 + 5] == 1
