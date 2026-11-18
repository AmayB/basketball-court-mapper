"""Calibration tests on synthetic renders with a known ground-truth homography."""
import cv2
import numpy as np

from hoopsvision.calibrate import (
    Calibrator,
    HomographySmoother,
    plausible,
    quad_correspondences,
    ransac_lines,
    split_line_families,
)
from hoopsvision.config import load_config
from hoopsvision.court import CourtSpec, apply_h

W, H = 640, 360


def broadcast_h() -> np.ndarray:
    """A sideline-camera-like view of the right half court."""
    court = np.array([[47, 0], [94, 0], [94, 50], [47, 50]], np.float32)
    img = np.array([[-60, 400], [700, 330], [520, 110], [80, 125]], np.float32)
    return cv2.getPerspectiveTransform(court, img).astype(np.float64)


def render(Hc: np.ndarray, court: CourtSpec) -> np.ndarray:
    """Wood floor, yellow lane with a dark-red border band, dark court lines,
    crowd-ish noise above the far sideline."""
    rng = np.random.default_rng(0)
    img = rng.integers(20, 90, (H, W, 3)).astype(np.uint8)  # crowd
    floor = apply_h(Hc, np.array([[0, 0], [94, 0], [94, 50], [0, 50]], float))
    cv2.fillPoly(img, [floor.round().astype(np.int32)], (92, 160, 205))  # wood (BGR)
    lc = court.lane_corners("right")
    lane = apply_h(Hc, np.array([lc["base_near"], lc["ft_near"], lc["ft_far"], lc["base_far"]]))
    cv2.fillPoly(img, [lane.round().astype(np.int32)], (40, 30, 150))      # border band
    inner = lane + (lane.mean(0) - lane) * 0.08
    cv2.fillPoly(img, [inner.round().astype(np.int32)], (40, 205, 240))    # lane fill
    for pl in court.polylines():
        uv = apply_h(Hc, pl)
        if np.isfinite(uv).all():
            cv2.polylines(img, [uv.round().astype(np.int32)], False, (30, 40, 60), 2, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_ransac_finds_two_line_families():
    rng = np.random.default_rng(1)
    t = rng.uniform(0, 200, 150)
    pts = np.concatenate([
        np.stack([t, 50 + 0.1 * t], 1), np.stack([t, 120 + 0.1 * t], 1),        # near-horizontal
        np.stack([20 + 0.7 * t[:80], t[:80]], 1), np.stack([160 + 0.7 * t[:80], t[:80]], 1),  # steep
        rng.uniform(0, 200, (40, 2)),                                             # clutter
    ])
    lines = ransac_lines(pts, n_lines=6, min_inliers=40)
    hz, st = split_line_families(lines)
    assert len(hz) == 2 and len(st) == 2


def test_quad_correspondences_orientation():
    court = CourtSpec()
    Hc = broadcast_h()
    lc = court.lane_corners("right")
    quad = apply_h(Hc, np.array([lc["base_near"], lc["ft_near"], lc["ft_far"], lc["base_far"]]))
    for order in ([2, 3, 0, 1], [3, 2, 1, 0]):  # any start corner, either winding
        src, dst = quad_correspondences(quad[order], court, "right")
        Hfit = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        assert np.allclose(apply_h(Hfit, src), apply_h(Hc, src), atol=1e-2)
    assert plausible(Hc, (H, W))


def test_calibrator_recovers_known_homography():
    cfg = load_config(None)
    court = CourtSpec.from_cfg(cfg.court)
    Hc = broadcast_h()
    frame = render(Hc, court)
    cal = Calibrator(cfg, (H, W), np.zeros((H, W), np.uint8))
    r = None
    for i in range(4):  # a few frames so the smoother settles
        r = cal.process(i, frame, np.zeros((0, 4)))
    assert r["H"] is not None and r["conf"] >= cfg.calibration.low_conf_threshold
    assert r["side"] == "right"
    # reprojection error over the visible lane / key area
    probe = np.array([[75, 17], [94, 33], [80, 25], [70, 10], [88, 40], [60, 25]], float)
    err = np.hypot(*(apply_h(r["H"], probe) - apply_h(Hc, probe)).T)
    assert err.max() < 6.0, err
    # and in court units (what metrics use): back-projected positions within ~1 ft
    uv = apply_h(Hc, probe)
    back = apply_h(np.linalg.inv(r["H"]), uv)
    assert np.hypot(*(back - probe).T).max() < 1.0


def test_smoother_follows_camera_motion_without_lag():
    Hc = broadcast_h()
    sm = HomographySmoother(alpha=0.3)
    sm.update(Hc, 1.0, (H, W))
    T = np.array([[1, 0, 12.0], [0, 1, -3.0], [0, 0, 1]])  # camera pans: image shifts
    sm.predict(T)
    Hn = T @ Hc
    sm.update(Hn, 1.0, (H, W))
    pts = np.array([[80, 25], [60, 10]], float)
    assert np.abs(apply_h(sm.current(), pts) - apply_h(Hn, pts)).max() < 1e-3
