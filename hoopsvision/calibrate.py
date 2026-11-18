"""Stage 3: dynamic camera calibration (court feet -> image pixels homography).

Generate-and-verify, per frame:

  observe   floor region (dominant floor colour, gated), court-line mask (thin
            bright OR dark structures inside the floor: arenas paint lines in
            either polarity), painted-lane candidate quadrilaterals.
  propose   (a) last smoothed H carried forward by camera motion (sparse LK flow
                on background features + RANSAC homography, or ECC),
            (b) one H per lane quad per basket side (4 corner correspondences),
            (c) manual anchors from config, if any.
  refine    ICP against the line mask: project ~1 ft-spaced template points of
            every court line, match each to its nearest line pixel (distance
            transform), re-fit H with RANSAC on (template -> pixel) pairs plus the
            lane corners; shrink the match radius each iteration.
  verify    score = fraction of visible template points that land within
            inlier_tol px of a detected line. Best-scoring hypothesis wins.
  smooth    motion-compensated EMA on the image positions of 4 reference court
            points (predict with camera motion, blend toward the measurement),
            so smoothing does not lag during pans; reset on cuts.
  confidence  from the line-agreement score; when too few template points are
            visible (tight zoom) the previous confidence decays per frame.
            Frames under calibration.low_conf_threshold are flagged and the
            metrics stage skips them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .court import CourtSpec, apply_h
from .video import iter_frames

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- observation

def floor_region(lab: np.ndarray, min_frac: float = 0.02) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (region mask, convex-hull region mask, floor Lab colour). The floor colour is the
    median of the lower-middle band; pixels close to it (lightness down-weighted
    for lighting falloff) form the floor; large components are unioned, closed and
    hole-filled so lines, paint and players inside the floor are included."""
    H, W = lab.shape[:2]
    band = lab[int(H * 0.6) : int(H * 0.9), int(W * 0.1) : int(W * 0.9)].reshape(-1, 3)
    fc = np.median(band, 0)
    d = np.linalg.norm((lab - fc) * np.array([0.35, 1.0, 1.0], np.float32), axis=2)
    fm = (d < 16).astype(np.uint8)
    fm = cv2.morphologyEx(fm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(fm)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_frac * H * W
    fm = keep[lbl].astype(np.uint8)
    fm = cv2.morphologyEx(fm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    region = _fill_holes(fm)
    hull = np.zeros_like(region)
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(hull, [cv2.convexHull(np.concatenate(cnts))], -1, 1, -1)
    return region, hull, fc


def _fill_holes(m: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out


def line_mask(gray: np.ndarray, region: np.ndarray, block: np.ndarray, thr: int = 14) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    m = ((th > thr) | (bh > thr)) & (cv2.erode(region, np.ones((5, 5), np.uint8)) > 0) & (block == 0)
    return m.astype(np.uint8)


def paint_candidates(lab: np.ndarray, region: np.ndarray, floor_lab: np.ndarray, block: np.ndarray,
                     paint_lab=None, tol: float = 22.0, max_quads: int = 4,
                     n_colors: int = 3) -> tuple[list[np.ndarray], np.ndarray]:
    """Lane quads (4x2, image px) and the union paint mask.

    With a known paint colour (config, or learned from a confident fit) the mask
    is a colour-distance threshold. Otherwise the saturated non-floor pixels in the
    floor hull are split into a few colour clusters (lanes are often bordered by
    a second colour) and every cluster plus the union proposes quads; geometric
    verification downstream picks the right one."""
    H, W = region.shape
    inside = (region > 0) & (block == 0)
    masks: list[np.ndarray] = []
    if paint_lab is not None:
        masks.append(np.linalg.norm(lab - np.asarray(paint_lab, np.float32), axis=2) < tol)
    else:
        d = np.linalg.norm((lab - floor_lab) * np.array([0.25, 1.0, 1.0], np.float32), axis=2)
        chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
        cand = (d > 14) & (chroma > 22) & inside
        masks.append(cand)
        ys, xs = np.nonzero(cand)
        if len(xs) > 200:
            from sklearn.cluster import KMeans

            sel = np.random.default_rng(0).choice(len(xs), min(3000, len(xs)), replace=False)
            ab = lab[ys[sel], xs[sel]][:, 1:]
            km = KMeans(n_colors, n_init=2, random_state=0).fit(ab)
            lbl = km.predict(lab[ys, xs][:, 1:])
            per = []
            for k in range(n_colors):
                mk = np.zeros((H, W), bool)
                mk[ys[lbl == k], xs[lbl == k]] = True
                per.append(mk)
            masks.extend(per)
            # lanes are often a fill colour plus a border band of another colour
            for i in range(n_colors):
                for j in range(i + 1, n_colors):
                    masks.append(per[i] | per[j])
    quads: list[np.ndarray] = []
    union = np.zeros((H, W), np.uint8)
    for mk in masks:
        m = _clean_paint_mask(mk & (region > 0), block)
        union |= m
        for q in _quads_from_mask(m, H * W, max_quads, block):
            if not any(np.abs(q - o).max() < 6 for o in quads):
                quads.append(q)
    return quads[: max_quads * 2], union


def _clean_paint_mask(m: np.ndarray, block: np.ndarray) -> np.ndarray:
    m = m.astype(np.uint8)
    m[block > 0] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


def _quads_from_mask(m: np.ndarray, img_area: int, max_quads: int, block: np.ndarray | None = None) -> list[np.ndarray]:
    """Quads from the whole mask (occluders split the lane into pieces, so pieces
    are fitted together) and from the largest component alone."""
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return []
    big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 0.0005 * img_area]
    if not big:
        return []
    merged = np.isin(lbl, big).astype(np.uint8)
    largest = (lbl == max(big, key=lambda i: st[i, cv2.CC_STAT_AREA])).astype(np.uint8)
    free = np.ones_like(m) if block is None else (block == 0).astype(np.uint8)
    out = []
    for comp in (merged, largest):
        area = int(comp.sum())
        if not (0.008 * img_area <= area <= 0.35 * img_area):
            continue
        q = quad_from_component(comp, block)
        if q is None:
            continue
        qmask = np.zeros_like(m)
        cv2.fillConvexPoly(qmask, q.round().astype(np.int32), 1)
        inter = float((qmask & comp & free).sum())
        union = float(((qmask | comp) & free).sum()) + 1e-9
        if inter / union < 0.6:
            continue
        if not any(np.abs(q - o).max() < 6 for o in out):
            out.append(q)
    return out[:max_quads]


def ransac_lines(pts: np.ndarray, n_lines: int = 6, tol: float = 1.5, min_inliers: int = 20,
                 iters: int = 400, rng=None) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Sequential RANSAC (vectorised over hypotheses): [(point, unit dir, n_inliers)]."""
    rng = rng or np.random.default_rng(0)
    pts = pts.astype(np.float64)
    found = []
    for _ in range(n_lines):
        if len(pts) < min_inliers:
            break
        ij = rng.integers(0, len(pts), (iters, 2))
        p0, p1 = pts[ij[:, 0]], pts[ij[:, 1]]
        d = p1 - p0
        L = np.hypot(d[:, 0], d[:, 1])
        ok = L >= 8
        if not ok.any():
            break
        p0, d, L = p0[ok], d[ok], L[ok]
        nrm = np.stack([-d[:, 1], d[:, 0]], 1) / L[:, None]
        dist = np.abs((pts[None, :, :] - p0[:, None, :]) @ nrm[:, :, None])[..., 0] if False else             np.abs((pts[None, :, 0] - p0[:, None, 0]) * nrm[:, None, 0] + (pts[None, :, 1] - p0[:, None, 1]) * nrm[:, None, 1])
        counts = (dist < tol).sum(1)
        k = int(np.argmax(counts))
        if counts[k] < min_inliers:
            break
        inl = dist[k] < tol
        sel = pts[inl].astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        found.append((np.array([x0, y0], float), np.array([vx, vy], float), int(inl.sum())))
        pts = pts[~inl]
    return found


def line_angle(d: np.ndarray) -> float:
    """Signed angle from horizontal in degrees, (-90, 90]."""
    a = np.degrees(np.arctan2(d[1], d[0]))
    if a <= -90:
        a += 180
    if a > 90:
        a -= 180
    return a


def split_line_families(lines, min_sep_deg: float = 15.0):
    """Split lines into (near-horizontal family, steeper family). Line angles live
    on a circle mod 180 deg, so two families are separated by the two largest
    circular gaps. Returns ([], []) if there is no clear two-family structure."""
    if len(lines) < 2:
        return [], []
    ang = np.array([line_angle(l[1]) for l in lines])
    order = np.argsort(ang)
    a = ang[order]
    gaps = np.append(np.diff(a), a[0] + 180 - a[-1])  # gaps[i] follows a[i]
    i, j = sorted(np.argsort(gaps)[-2:])
    if min(gaps[i], gaps[j]) < min_sep_deg:
        return [], []
    fam_a = [lines[k] for k in order[i + 1 : j + 1]]
    fam_b = [lines[k] for k in np.concatenate([order[j + 1 :], order[: i + 1]])]

    def horizness(fam):
        return np.mean([np.cos(np.radians(line_angle(l[1]))) for l in fam])

    return (fam_a, fam_b) if horizness(fam_a) > horizness(fam_b) else (fam_b, fam_a)


def _intersect(l1, l2):
    (p1, d1), (p2, d2) = l1[:2], l2[:2]
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def quad_from_component(comp: np.ndarray, block: np.ndarray | None) -> np.ndarray | None:
    """Fit a lane quad to one paint component: RANSAC lines on its boundary
    (ignoring boundary created by occluding players or the image border), then
    pick two near-horizontal lines (lane sides) and two steeper ones (baseline /
    free-throw line) and intersect them."""
    H, W = comp.shape
    edge = cv2.morphologyEx(comp, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if block is not None:
        edge &= cv2.dilate(block.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0
    edge[:3, :] = edge[-3:, :] = False
    edge[:, :3] = edge[:, -3:] = False
    ys, xs = np.nonzero(edge)
    if len(xs) < 60:
        return None
    lines = ransac_lines(np.stack([xs, ys], 1), n_lines=8, min_inliers=max(12, len(xs) // 40), iters=500)
    hz, st = split_line_families(lines)
    if len(hz) < 2 or len(st) < 2:
        return None
    free = np.ones_like(comp) if block is None else (block == 0).astype(np.uint8)
    hz = sorted(hz, key=lambda l: -l[2])[:3]
    st = sorted(st, key=lambda l: -l[2])[:3]
    best, best_iou = None, 0.0
    from itertools import combinations

    for h1, h2 in combinations(hz, 2):
        for s1, s2 in combinations(st, 2):
            pts = [_intersect(h, v) for h in (h1, h2) for v in (s1, s2)]
            if any(p is None for p in pts):
                continue
            q = cv2.convexHull(np.array(pts, np.float32)).reshape(-1, 2)
            if len(q) != 4 or (np.abs(q[:, 0] - W / 2) > W).any() or (np.abs(q[:, 1] - H / 2) > H).any():
                continue
            qm = np.zeros_like(comp)
            cv2.fillConvexPoly(qm, q.round().astype(np.int32), 1)
            iou = float((qm & comp & free).sum()) / (float(((qm | comp) & free).sum()) + 1e-9)
            if iou > best_iou:
                best, best_iou = q.astype(np.float64), iou
    return best


def fit_quad(contour: np.ndarray) -> np.ndarray | None:
    """Fit a quadrilateral to a contour: approxPolyDP on the hull for the corners,
    then refine each side with a least-squares line through nearby contour points
    and intersect adjacent sides (robust to rounded/occluded corners)."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    approx = None
    for eps in (0.02, 0.03, 0.04, 0.06, 0.08):
        a = cv2.approxPolyDP(hull, eps * peri, True)
        if len(a) == 4:
            approx = a.reshape(4, 2).astype(np.float64)
            break
    if approx is None:
        return None
    pts = contour.reshape(-1, 2).astype(np.float64)
    lines = []
    for i in range(4):
        p, q = approx[i], approx[(i + 1) % 4]
        d = q - p
        L = np.hypot(*d)
        if L < 5:
            return None
        n = np.array([-d[1], d[0]]) / L
        t = (pts - p) @ (d / L)
        near = (np.abs((pts - p) @ n) < 4) & (t > 0.1 * L) & (t < 0.9 * L)
        sel = pts[near] if near.sum() >= 8 else np.stack([p, q])
        vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        s = np.linalg.solve(A, p2 - p1)
        out.append(p1 + s[0] * d1)
    return np.array(out)


def quad_correspondences(quad: np.ndarray, court: CourtSpec, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Map image lane quad corners to court lane corners for a basket side.
    The pair of opposite edges closer to horizontal are the lane sides (parallel
    to the court's long axis in a sideline broadcast view); the upper one is the
    far side; along each side, the left image endpoint has the smaller court x."""
    e = [(quad[i], quad[(i + 1) % 4]) for i in range(4)]

    def horiz(p, q):
        d = q - p
        return abs(d[0]) / (np.hypot(*d) + 1e-9)

    pair = (0, 2) if horiz(*e[0]) + horiz(*e[2]) >= horiz(*e[1]) + horiz(*e[3]) else (1, 3)
    s1, s2 = e[pair[0]], e[pair[1]]
    far, near = (s1, s2) if (s1[0][1] + s1[1][1]) < (s2[0][1] + s2[1][1]) else (s2, s1)
    far = sorted(far, key=lambda p: p[0])
    near = sorted(near, key=lambda p: p[0])
    c = court.lane_corners(side)
    if side == "right":  # left image endpoint = free-throw end
        court_pts = [c["ft_far"], c["base_far"], c["ft_near"], c["base_near"]]
    else:
        court_pts = [c["base_far"], c["ft_far"], c["base_near"], c["ft_near"]]
    img_pts = [far[0], far[1], near[0], near[1]]
    return np.array(court_pts, np.float64), np.array(img_pts, np.float64)


def basket_side_hint(quad: np.ndarray) -> str | None:
    """Lean of the lane's end lines: in a sideline view, lines perpendicular to
    the sideline lean like '\\' on the right half and '/' on the left half."""
    e = [quad[(i + 1) % 4] - quad[i] for i in range(4)]
    hz = [abs(d[0]) / (np.hypot(*d) + 1e-9) for d in e]
    ends = (1, 3) if hz[0] + hz[2] >= hz[1] + hz[3] else (0, 2)
    lean = 0.0
    for i in ends:
        d = e[i] if e[i][1] > 0 else -e[i]  # orient downward
        lean += d[0] / (np.hypot(*d) + 1e-9)
    if abs(lean) < 0.1:
        return None
    return "right" if lean > 0 else "left"


# ------------------------------------------------------------- camera motion

class MotionEstimator:
    """Frame-to-frame homography of the background (camera pan/tilt/zoom)."""

    def __init__(self, method: str = "lk"):
        self.method = method
        self.prev = None

    def reset(self):
        self.prev = None

    def __call__(self, gray: np.ndarray, block: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns (M prev->cur, quality in [0,1])."""
        prev, self.prev = self.prev, gray
        if prev is None:
            return np.eye(3), 0.0
        mask = (block == 0).astype(np.uint8) * 255
        if self.method == "ecc":
            return self._ecc(prev, gray, mask)
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=500, qualityLevel=0.01, minDistance=7, mask=mask)
        if p0 is None or len(p0) < 12:
            return np.eye(3), 0.0
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < 12:
            return np.eye(3), 0.0
        M, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 2.0)
        if M is None:
            return np.eye(3), 0.0
        return M, float(inl.mean())

    def _ecc(self, prev, cur, mask):
        s = 0.5
        a = cv2.resize(prev, None, fx=s, fy=s).astype(np.float32)
        b = cv2.resize(cur, None, fx=s, fy=s).astype(np.float32)
        m = cv2.resize(mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        W = np.eye(3, dtype=np.float32)
        try:
            cc, W = cv2.findTransformECC(a, b, W, cv2.MOTION_HOMOGRAPHY,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5), m, 5)
        except cv2.error:
            return np.eye(3), 0.0
        S = np.diag([s, s, 1.0])
        # ECC returns a warp mapping cur -> prev coordinates; invert for prev -> cur
        return np.linalg.inv(np.linalg.inv(S) @ W.astype(np.float64) @ S), float(cc)


# ------------------------------------------------------------------ fitting

@dataclass
class Fit:
    H: np.ndarray
    score: float          # line agreement in [0,1]
    n_visible: int
    source: str           # paint | propagated | manual
    side: str | None


class LineMatcher:
    """Nearest-line-pixel lookup and agreement scoring for one frame."""

    def __init__(self, lines: np.ndarray, valid: np.ndarray):
        self.valid = valid
        inv = (lines == 0).astype(np.uint8)
        self.dist, labels = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(lines)
        # label k (1-based) corresponds to the k-th zero pixel of `inv` in scan order
        self.lut = np.zeros((labels.max() + 1, 2), np.float32)
        if len(xs):
            zero_order = labels[ys, xs]
            self.lut[zero_order] = np.stack([xs, ys], 1)
        self.labels = labels
        self.has_lines = len(xs) > 0

    def visible(self, uv: np.ndarray) -> np.ndarray:
        H, W = self.valid.shape
        ok = np.isfinite(uv).all(1)
        u = np.where(ok, uv[:, 0], -1).round().astype(int)
        v = np.where(ok, uv[:, 1], -1).round().astype(int)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ok[ok] &= self.valid[v[ok], u[ok]] > 0
        return ok

    def query(self, uv: np.ndarray, vis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = uv[vis, 0].round().astype(int)
        v = uv[vis, 1].round().astype(int)
        return self.dist[v, u], self.lut[self.labels[v, u]]

    def score(self, H: np.ndarray, tpl: np.ndarray, tol: float) -> tuple[float, int]:
        uv = apply_h(H, tpl)
        vis = self.visible(uv)
        n = int(vis.sum())
        if n == 0 or not self.has_lines:
            return 0.0, n
        d, _ = self.query(uv, vis)
        return float((d <= tol).mean()), n


def icp_refine(H0: np.ndarray, tpl: np.ndarray, matcher: LineMatcher, radii, anchors=None) -> np.ndarray:
    """Iteratively re-fit H on (template point -> nearest line pixel) pairs."""
    H = H0
    for r in radii:
        uv = apply_h(H, tpl)
        vis = matcher.visible(uv)
        if vis.sum() < 8:
            break
        d, nn = matcher.query(uv, vis)
        good = d <= r
        src, dst = tpl[vis][good], nn[good]
        if anchors is not None:
            rep = max(4, len(src) // 20)  # keep the lane corners influential
            src = np.concatenate([src] + [anchors[0]] * rep)
            dst = np.concatenate([dst] + [anchors[1]] * rep)
        if len(src) < 8:
            break
        Hn, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), cv2.RANSAC, max(r / 2, 2.0))
        if Hn is None or not plausible(Hn, matcher.valid.shape):
            break
        H = Hn
    return H


def plausible(H: np.ndarray, shape) -> bool:
    """Reject degenerate homographies: the court must project to a convex quad
    with the far sideline above the near one, at a sane scale."""
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        return False
    Hh, W = shape[:2]
    c = apply_h(H, np.array([[0, 0], [94, 0], [94, 50], [0, 50]], float))
    if not np.isfinite(c).all():
        return False
    if not cv2.isContourConvex(c.astype(np.float32).reshape(-1, 1, 2)):
        return False
    near_y = (c[0, 1] + c[1, 1]) / 2
    far_y = (c[2, 1] + c[3, 1]) / 2
    if not far_y < near_y:
        return False
    # 1 ft at the lane must be between 0.5 px and 1/3 of the frame width
    lane = apply_h(H, np.array([[75, 25], [76, 25], [19, 25], [18, 25]], float))
    s = min(np.hypot(*(lane[1] - lane[0])), np.hypot(*(lane[3] - lane[2])))
    return 0.5 < s < W / 3


# ---------------------------------------------------------------- smoothing

class HomographySmoother:
    """Motion-compensated EMA over the image positions of 4 reference court points."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.X = None   # 4x2 court reference points
        self.P = None   # 4x2 smoothed image positions

    def reset(self):
        self.X = self.P = None

    def predict(self, M: np.ndarray):
        if self.P is not None:
            self.P = apply_h(M, self.P)

    def current(self) -> np.ndarray | None:
        if self.P is None or not np.isfinite(self.P).all():
            return None
        return cv2.getPerspectiveTransform(self.X.astype(np.float32), self.P.astype(np.float32)).astype(np.float64)

    def update(self, H: np.ndarray, weight: float, shape, jump_px: float = 25.0):
        h, w = shape[:2]
        if self.X is None or self._stale(w, h):
            q = np.array([[0.25 * w, 0.55 * h], [0.75 * w, 0.55 * h], [0.75 * w, 0.9 * h], [0.25 * w, 0.9 * h]])
            self.X = apply_h(np.linalg.inv(H), q)
            self.P = q.copy()
            return
        meas = apply_h(H, self.X)
        a = float(np.clip(self.alpha * weight, 0.0, 1.0))
        if np.median(np.hypot(*(meas - self.P).T)) > jump_px and weight > 0.8:
            self.P = meas  # confident re-anchor: don't drag a stale estimate along
        else:
            self.P = a * meas + (1 - a) * self.P

    def _stale(self, w, h) -> bool:
        P = self.P
        return P is None or not np.isfinite(P).all() or (np.abs(P[:, 0] - w / 2) > 1.5 * w).any() or \
            (np.abs(P[:, 1] - h / 2) > 1.5 * h).any()


# ------------------------------------------------------------- static overlays

def static_overlay_mask(video: str, start: int, end: int, shape, n_samples: int = 40) -> np.ndarray:
    """Broadcast graphics (scorebug, watermark) stay fixed while the camera pans:
    low temporal std of the gradient over sampled frames inside the frame."""
    H, W = shape[:2]
    step = max(1, (end - start) // n_samples)
    stack = []
    for fi, fr in iter_frames(video, start, end - start):
        if (fi - start) % step:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (W // 2, H // 2)).astype(np.float32)
        mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        stack.append(mag)
    if len(stack) < 5:
        return np.zeros((H, W), np.uint8)
    s = np.stack(stack)
    strong = s.mean(0) > 40       # consistently textured ...
    still = s.std(0) < 0.35 * (s.mean(0) + 1)  # ... and that texture does not move
    m = (strong & still).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 20
    m = keep[lbl].astype(np.uint8)
    # graphics show up as outlines + text: fill enclosed areas, keep thin links thin
    out = _fill_holes(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)))
    out = cv2.dilate(out, np.ones((3, 3), np.uint8))
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST)


# ------------------------------------------------------------------- driver

class Calibrator:
    def __init__(self, cfg, shape, overlay: np.ndarray):
        self.c = cfg.calibration
        self.court = CourtSpec.from_cfg(cfg.court)
        self.tpl = self.court.sample_points(self.c.template_spacing_ft)
        self.shape = shape
        self.overlay = overlay
        self.motion = MotionEstimator(self.c.motion_method)
        self.smoother = HomographySmoother(self.c.smoothing_alpha)
        self.conf = 0.0
        self.since_anchor = 10**9
        self.paint_lab = self.c.paint_lab
        self.side: str | None = None if self.c.basket_side == "auto" else self.c.basket_side
        self.manual = {int(a["frame"]): np.asarray(a["points"], float) for a in (self.c.manual_anchors or [])}

    def reset(self):
        self.motion.reset()
        self.smoother.reset()
        self.conf = 0.0
        self.since_anchor = 10**9
        if self.c.basket_side == "auto":
            self.side = None

    def block_mask(self, boxes: np.ndarray) -> np.ndarray:
        H, W = self.shape[:2]
        b = self.overlay.copy()
        for x1, y1, x2, y2 in boxes:
            pad = 0.08 * (y2 - y1)
            b[max(0, int(y1 - pad)) : int(y2 + pad), max(0, int(x1 - pad)) : int(x2 + pad)] = 1
        for x0, y0, x1, y1 in self.c.ignore_regions or []:
            b[int(y0 * H) : int(y1 * H), int(x0 * W) : int(x1 * W)] = 1
        return b

    # -- verification helpers ------------------------------------------------
    def lane_iou(self, H: np.ndarray, side: str | None, paint: np.ndarray | None, block: np.ndarray):
        """IoU between the projected lane polygon and the observed paint, over
        unoccluded pixels. None when the lane is not visibly in frame."""
        if side is None or paint is None:
            return None
        Hh, W = self.shape[:2]
        lc = self.court.lane_corners(side)
        uv = apply_h(H, np.array([lc["base_near"], lc["ft_near"], lc["ft_far"], lc["base_far"]]))
        if not np.isfinite(uv).all():
            return None
        uv = np.clip(uv, -4 * W, 5 * W)
        qm = np.zeros((Hh, W), np.uint8)
        cv2.fillConvexPoly(qm, uv.round().astype(np.int32), 1)
        free = (block == 0).astype(np.uint8)
        if (qm & free).sum() < 0.012 * Hh * W:
            return None
        # every paint blob the projected lane touches counts in full, so paint
        # spilling outside the polygon (a misplaced lane) is penalised
        n, lbl = cv2.connectedComponents(paint & free)
        touched = np.unique(lbl[(qm > 0) & (lbl > 0)])
        pm = np.isin(lbl, touched).astype(np.uint8) if len(touched) else np.zeros_like(qm)
        inter = float((qm & pm).sum())
        union = float(((qm & free) | pm).sum()) + 1e-9
        return inter / union

    def known_paint_mask(self, lab: np.ndarray, region: np.ndarray, floor_lab: np.ndarray) -> np.ndarray | None:
        """Pixels near the learned paint colour AND clearly closer to it than to
        this frame's floor colour (exposure flashes pull floor and paint together)."""
        if self.paint_lab is None:
            return None
        dp = np.linalg.norm(lab - np.asarray(self.paint_lab, np.float32), axis=2)
        df = np.linalg.norm(lab - np.asarray(floor_lab, np.float32), axis=2)
        m = (dp < self.c.paint_lab_tol) & (dp < 0.6 * df)
        m = (m & (region > 0)).astype(np.uint8)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def learn_paint(self, H, side, lab, block, lines):
        """Learn the lane fill colour from a confident fit (EMA, so it follows
        exposure changes). Used for the per-frame lane-consistency check."""
        lc = self.court.lane_corners(side)
        uv = apply_h(H, np.array([lc["base_near"], lc["ft_near"], lc["ft_far"], lc["base_far"]]))
        qm = np.zeros(self.shape[:2], np.uint8)
        cv2.fillConvexPoly(qm, np.clip(uv, -1e4, 1e4).round().astype(np.int32), 1)
        qm = cv2.erode(qm, np.ones((7, 7), np.uint8))  # stay off the lane border band
        sel = (qm > 0) & (block == 0) & (lines == 0)
        if sel.sum() < 200:
            return
        med = np.median(lab[sel], 0)
        self.paint_lab = med if self.paint_lab is None else 0.5 * np.asarray(self.paint_lab) + 0.5 * med

    def confidence(self, score: float, n_vis: int, iou: float | None) -> float:
        c = self.c
        if n_vis < c.min_visible_template_pts:
            line_conf = 0.0
        else:
            line_conf = float(np.clip((score - 0.15) / (0.5 - 0.15), 0.0, 1.0))
        if iou is None:
            return 0.8 * line_conf  # no independent check available
        lane_conf = float(np.clip((iou - 0.35) / (0.75 - 0.35), 0.0, 1.0))
        return min(0.4 * line_conf + 0.6 * lane_conf, 0.25 + lane_conf)

    def process(self, fi: int, frame: np.ndarray, boxes: np.ndarray) -> dict:
        c = self.c
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        block = self.block_mask(boxes)
        # equalised gray keeps optical flow stable across exposure flashes
        M, mq = self.motion(cv2.equalizeHist(gray), block)
        self.smoother.predict(M)

        region, hull, floor_lab = floor_region(lab)
        Hprop = self.smoother.current()
        search = (
            fi % max(1, int(c.paint_search_every)) == 0
            or Hprop is None
            or self.conf < c.low_conf_threshold
            or fi in self.manual
        )
        if search:
            # colour clustering (not the learned colour) so a two-tone lane is not shrunk
            quads, cand_paint = paint_candidates(lab, hull, floor_lab, block, c.paint_lab, c.paint_lab_tol)
        else:
            quads, cand_paint = [], np.zeros_like(region)
        known = self.known_paint_mask(lab, hull, floor_lab)
        paint_for_check = known if known is not None else (cand_paint if search else None)
        region = np.maximum(region, cand_paint)
        lines = line_mask(gray, region, block)
        # the lane boundary is a court line even when it is a colour edge, not a painted stripe
        edge = cv2.morphologyEx(cand_paint, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        lines = np.maximum(lines, edge & (block == 0).astype(np.uint8))
        valid = (cv2.dilate(region, np.ones((9, 9), np.uint8)) > 0) & (block == 0)
        matcher = LineMatcher(lines, valid.astype(np.uint8))

        cands: list = []

        def add(fit: Fit):
            iou = self.lane_iou(fit.H, fit.side, paint_for_check, block)
            cands.append((self.confidence(fit.score, fit.n_visible, iou), fit, iou))

        if Hprop is not None and plausible(Hprop, self.shape):
            Hr = icp_refine(Hprop, self.tpl, matcher, c.icp_radius_px[-2:])
            s, n = matcher.score(Hr, self.tpl, c.inlier_tol_px)
            add(Fit(Hr, s, n, "propagated", self.side))
        sides = ["left", "right"] if c.basket_side == "auto" else [c.basket_side]
        for q in quads:
            hint = basket_side_hint(q)
            q_sides = [hint] if (hint is not None and c.basket_side == "auto") else sides
            for side in q_sides:
                src, dst = quad_correspondences(q, self.court, side)
                H0 = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)
                if not plausible(H0, self.shape):
                    continue
                Hr = icp_refine(H0, self.tpl, matcher, c.icp_radius_px, anchors=(src, dst))
                s, n = matcher.score(Hr, self.tpl, c.inlier_tol_px)
                add(Fit(Hr, s, n, "paint", side))
        if fi in self.manual:
            pts = self.manual[fi]
            Hm, _ = cv2.findHomography(pts[:, 2:4], pts[:, 0:2], 0)
            cands.append((1.0, Fit(Hm, 1.0, len(self.tpl), "manual", self.side), None))

        conf, best, iou = max(cands, key=lambda t: t[0], default=(0.0, None, None))
        # A propagated fit that the lane check could not verify only has line
        # agreement, which ICP can partially "earn" by snapping to player edges.
        # Cap it by time since the last verified anchor so drift gets flagged.
        if best is not None and best.source == "propagated" and (iou is None or iou < 0.6):
            cap = max(0.0, 1.0 - self.since_anchor / float(c.max_propagation_frames))
            conf = min(conf, cap)
        # Tight zoom / lane out of view: no evidence either way, so carry the
        # motion-propagated estimate with decaying confidence. If the lane IS in
        # view and disagrees (low IoU), that is evidence against: no carry-over.
        evidence_against = iou is not None and iou < 0.35
        if Hprop is not None and not evidence_against and self.since_anchor < c.max_propagation_frames:
            prop_conf = self.conf * c.propagation_decay * (0.5 + 0.5 * mq)
            if prop_conf > conf:
                conf = prop_conf
                best = Fit(Hprop, best.score if best else 0.0, best.n_visible if best else 0, "propagated", self.side)

        if best is not None and conf > 0:
            self.smoother.update(best.H, conf, self.shape)
            verified = best.source in ("paint", "manual") or (iou is not None and iou >= 0.6)
            if verified and conf >= c.low_conf_threshold:
                self.since_anchor = 0
                self.side = best.side or self.side
                if conf > 0.8:
                    self.learn_paint(best.H, self.side, lab, block, lines)
            else:
                self.since_anchor += 1
        else:
            self.since_anchor += 1
        self.conf = conf
        Hs = self.smoother.current()
        if Hs is None or not plausible(Hs, self.shape):
            Hs, conf = None, 0.0
            self.conf = 0.0
        return {
            "frame": fi, "H": Hs, "H_raw": best.H if best is not None else None, "conf": float(conf),
            "source": best.source if best else "none",
            "side": self.side,
            "score": best.score if best else 0.0,
            "lane_iou": iou,
            "n_visible": best.n_visible if best else 0,
            "motion_q": mq, "n_quads": len(quads),
        }


def run_calibration(video: str, cfg, start: int, end: int, segments, dets) -> dict:
    from .video import probe

    info = probe(video)
    shape = (info.height, info.width)
    overlay = static_overlay_mask(video, start, end, shape) if cfg.calibration.static_overlay_detect else \
        np.zeros(shape, np.uint8)
    cal = Calibrator(cfg, shape, overlay)
    persons = dets[dets.cls == "person"]
    boxes_by_frame = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in persons.groupby("frame")}
    seg_starts = {a for a, _ in segments}
    n = end - start
    Hs = np.full((n, 3, 3), np.nan)
    conf = np.zeros(n)
    meta = []
    for fi, frame in iter_frames(video, start, n):
        if fi in seg_starts:
            cal.reset()
        r = cal.process(fi, frame, boxes_by_frame.get(fi, np.zeros((0, 4))))
        if r["H"] is not None:
            Hs[fi - start] = r["H"]
        conf[fi - start] = r["conf"]
        meta.append({k: v for k, v in r.items() if k not in ("H", "H_raw")})
        if r["H"] is not None and r["H_raw"] is not None:
            q = np.array([[0.3 * shape[1], 0.7 * shape[0]], [0.7 * shape[1], 0.7 * shape[0]]])
            a = apply_h(r["H"], apply_h(np.linalg.inv(r["H_raw"]), q))
            meta[-1]["smooth_dev_px"] = float(np.nanmax(np.hypot(*(a - q).T)))
        if fi % 100 == 0:
            log.info("calibrate frame %d: conf=%.2f src=%s side=%s score=%.2f", fi, r["conf"], r["source"],
                     r["side"], r["score"])
    return {"H": Hs, "conf": conf, "overlay": overlay, "meta": meta}
