import cv2
import numpy as np
import pandas as pd

from hoopsvision.config import load_config
from hoopsvision.detect import box_iou, match_pose
from hoopsvision.scenecut import detect_cuts, hist_distance, hist_feature, segments_from_cuts
from hoopsvision.track import run_tracking


def test_hist_distance_detects_shot_change():
    a = np.full((90, 160, 3), (40, 120, 200), np.uint8)
    b = a.copy(); b[:5] += 3  # tiny change: same shot
    c = np.full((90, 160, 3), (200, 30, 30), np.uint8)
    assert hist_distance(hist_feature(a), hist_feature(b)) < 0.1
    assert hist_distance(hist_feature(a), hist_feature(c)) > 0.9


def test_detect_cuts_merges_close_peaks():
    d = np.zeros(100); d[30] = 0.8; d[32] = 0.9; d[70] = 0.6
    assert detect_cuts(d, 0.5, 10) == [32, 70]
    assert segments_from_cuts(0, 100, [32, 70]) == [(0, 32), (32, 70), (70, 100)]


def test_pose_matching_hungarian():
    persons = np.array([[0, 0, 10, 20], [50, 0, 60, 20]], float)
    poses = np.array([[51, 1, 61, 21], [100, 100, 110, 120]], float)
    kp = np.stack([np.full((17, 3), 1.0), np.full((17, 3), 2.0)])
    out = match_pose(persons, poses, kp, 0.4)
    assert np.isnan(out[0]).all() and (out[1] == 1.0).all()
    assert box_iou(persons, persons).diagonal().round(6).tolist() == [1.0, 1.0]


def _synthetic_dets(n=60, occlude=range(20, 30)):
    rows = []
    for f in range(n):
        for k, x0 in enumerate((100, 300)):
            if k == 0 and f in occlude:
                continue  # player 0 hidden behind a screen for 10 frames
            x = x0 + 2 * f
            rows.append(dict(frame=f, det_id=k, cls="person", x1=x, y1=100, x2=x + 30, y2=180, conf=0.9, kpts=None))
    return pd.DataFrame(rows)


def test_tracking_survives_occlusion_and_resets_on_cut():
    cfg = load_config(None, {"tracking": {"min_track_length": 3}})
    dets = _synthetic_dets()
    tr = run_tracking(dets, [(0, 60)], cfg, fps=30)
    ids_p0 = tr[tr.det_id == 0].track_id.unique()
    assert len(ids_p0) == 1, "occluded player should keep its ID"
    tr2 = run_tracking(dets, [(0, 40), (40, 60)], cfg, fps=30)
    before = set(tr2[tr2.frame < 40].track_id); after = set(tr2[tr2.frame >= 40].track_id)
    assert before and after and before.isdisjoint(after), "IDs must not span a cut"


def test_exposure_flash_is_not_a_cut_but_new_view_is():
    from hoopsvision.scenecut import is_cut, struct_corr, struct_feature
    cfg = load_config(None).scenecut
    rng = np.random.default_rng(1)
    base = (rng.uniform(0, 1, (90, 160)) > 0.97).astype(np.uint8) * 200 + 30
    base = cv2.GaussianBlur(np.dstack([base] * 3).astype(np.uint8), (5, 5), 0)
    washed = cv2.convertScaleAbs(base, alpha=0.5, beta=120)  # global exposure change
    other = np.ascontiguousarray(base[::-1, ::-1])           # different structure
    fa, fw, fo = struct_feature(base), struct_feature(washed), struct_feature(other)
    hd = hist_distance(hist_feature(base), hist_feature(washed))
    assert not is_cut(hd, struct_corr(fa, fw), cfg)
    assert is_cut(0.2, struct_corr(fa, fo), cfg)
