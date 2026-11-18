"""Stage 6: compose the 720x900 dashboard video.

  header        title + subtitle + team legend
  broadcast     the frame with team-coloured skeletons, foot ellipses and speed pills
  tactical      top-down court: Voronoi control regions, player dots, attacking pill
  tiles         Space Control (frontcourt %), Team Spacing (hull area),
                Distance Covered, Open Space (offense gap to nearest defender)

Rendering only reads cached tables (tracks, frame metrics, detections,
homographies), so it can be re-run without the models. Text uses a bundled
font (DejaVu Sans Bold from matplotlib) so output is identical across machines.
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from PIL import Image, ImageDraw, ImageFont

from .config import hex_to_bgr, hex_to_rgb
from .court import FT_TO_M, CourtSpec
from .detect import kpts_from_row
from .metrics import FT2_TO_M2, voronoi_cells
from .video import VideoWriter, iter_frames, probe

log = logging.getLogger(__name__)

# text tokens (text never wears a series colour)
INK = (241, 245, 249)
INK_2 = (148, 163, 184)
INK_3 = (100, 116, 139)
PANEL = (17, 26, 46)
LINE = (51, 65, 85)

COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
              (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    import matplotlib

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", name), size)


class Layout:
    def __init__(self, W: int, Hh: int, src_w: int, src_h: int):
        self.W, self.H = W, Hh
        self.header = (0, 0, W, 58)
        bh = int(round(W * src_h / src_w))
        self.broadcast = (0, 58, W, 58 + bh)
        top = 58 + bh
        self.tiles_h = 128
        self.tactical = (0, top, W, Hh - self.tiles_h)
        self.tiles = (0, Hh - self.tiles_h, W, Hh)


class CourtCanvas:
    """Static top-down court (navy) with a ft -> px transform."""

    def __init__(self, court: CourtSpec, box, bg_bgr, pad=14, title_h=34):
        x0, y0, x1, y1 = box
        avail_w, avail_h = x1 - x0 - 2 * pad, y1 - y0 - title_h - pad
        self.s = min(avail_w / court.length, avail_h / court.width)
        cw, ch = court.length * self.s, court.width * self.s
        self.ox = x0 + (x1 - x0 - cw) / 2
        self.oy = y0 + title_h + (avail_h - ch) / 2
        self.court = court
        self.bg = bg_bgr
        self.rect = (int(self.ox), int(self.oy), int(self.ox + cw), int(self.oy + ch))

    def px(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(xy)
        # y=0 (near sideline) at the bottom, matching the broadcast view
        return np.stack([self.ox + xy[:, 0] * self.s, self.oy + (self.court.width - xy[:, 1]) * self.s], 1)

    def draw_lines(self, img):
        for pl in self.court.polylines():
            p = (self.px(pl) * 16).round().astype(np.int32)
            cv2.polylines(img, [p], False, (150, 120, 95), 1, cv2.LINE_AA, shift=4)
        hoops = [self.court.hoop(s) for s in ("left", "right")]
        for h in hoops:
            c = (self.px(h)[0] * 16).round().astype(int)
            cv2.circle(img, tuple(c), int(0.75 * self.s * 16), (150, 120, 95), 1, cv2.LINE_AA, shift=4)


class Smoother:
    """Displayed tile values: EMA over valid frames; holds and greys out when stale."""

    def __init__(self, alpha: float, stale_after: int):
        self.a = alpha
        self.v: dict[str, float] = {}
        self.age: dict[str, int] = {}
        self.stale_after = stale_after

    def push(self, key: str, x):
        if x is None or not np.isfinite(x):
            self.age[key] = self.age.get(key, 10**6) + 1
            return
        self.v[key] = x if key not in self.v else self.a * x + (1 - self.a) * self.v[key]
        self.age[key] = 0

    def get(self, key):
        return self.v.get(key)

    def stale(self, key) -> bool:
        return self.age.get(key, 10**6) > self.stale_after


def _pill(draw: ImageDraw.ImageDraw, xy, text, font, fill, ink=INK, pad=(6, 2), anchor="mb"):
    x, y = xy
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    w, h = r - l + 2 * pad[0], b - t + 2 * pad[1]
    if anchor == "mb":
        x0, y0 = x - w / 2, y - h
    else:  # left-top
        x0, y0 = x, y
    draw.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=h / 2, fill=fill)
    draw.text((x0 + pad[0] - l, y0 + pad[1] - t), text, font=font, fill=ink)
    return w, h


def _split_bar(img, x, y, w, h, frac_a, col_a, col_b, gap=2):
    """Two-segment bar with a 2 px surface gap and rounded outer ends."""
    if frac_a is None or not np.isfinite(frac_a):
        cv2.rectangle(img, (x, y), (x + w, y + h), LINE, -1)
        return
    frac_a = float(np.clip(frac_a, 0.0, 1.0))
    wa = int(round((w - gap) * frac_a))
    r = h // 2
    for (x0, x1, col) in ((x, x + wa, col_a), (x + wa + gap, x + w, col_b)):
        if x1 - x0 < 1:
            continue
        cv2.rectangle(img, (x0 + r, y), (max(x0 + r, x1 - r), y + h - 1), col, -1)
        cv2.circle(img, (x0 + r, y + r), r, col, -1, cv2.LINE_AA)
        cv2.circle(img, (max(x0 + r, x1 - r), y + r), r, col, -1, cv2.LINE_AA)
        # square the inner (gap-side) end
        if col is col_a:
            cv2.rectangle(img, (max(x0, x1 - r), y), (x1 - 1, y + h - 1), col, -1)
        else:
            cv2.rectangle(img, (x0, y), (min(x1, x0 + r), y + h - 1), col, -1)


def render_dashboard(video, out_path, tracks: pd.DataFrame, fm: pd.DataFrame, dets: pd.DataFrame,
                     H_all: np.ndarray, cfg, start: int, end: int):
    rc = cfg.render
    info = probe(video)
    W, Hh = int(rc.width), int(rc.height)
    L = Layout(W, Hh, info.width, info.height)
    court = CourtSpec.from_cfg(cfg.court)
    labels = list(cfg.teams.labels)
    team_bgr = [hex_to_bgr(c) for c in rc.team_colors]
    team_rgb = [hex_to_rgb(c) for c in rc.team_colors]
    ref_bgr, ref_rgb = hex_to_bgr(rc.referee_color), hex_to_rgb(rc.referee_color)
    bg_bgr = hex_to_bgr(rc.background)
    court_bg = hex_to_bgr(rc.court_bg)
    thr = cfg.calibration.low_conf_threshold
    sx = W / info.width

    f_title, f_sub, f_small, f_tile_t, f_val, f_lbl = (_font(22), _font(11, False), _font(11), _font(10),
                                                       _font(17), _font(10, False))
    canvas = CourtCanvas(court, L.tactical, court_bg)

    # static background (header, panels, court) rendered once
    base = np.zeros((Hh, W, 3), np.uint8)
    base[:] = bg_bgr
    x0, y0, x1, y1 = canvas.rect
    cv2.rectangle(base, (x0 - 8, y0 - 8), (x1 + 8, y1 + 8), court_bg, -1)
    canvas.draw_lines(base)
    tw = (W - 16 * 2 - 8 * 3) // 4
    tile_boxes = [(16 + i * (tw + 8), L.tiles[1] + 8, 16 + i * (tw + 8) + tw, Hh - 12) for i in range(4)]
    for bx in tile_boxes:
        cv2.rectangle(base, bx[:2], bx[2:], PANEL[::-1], -1)
    pil = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.text((16, 9), rc.title, font=f_title, fill=INK)
    d.text((16, 37), rc.subtitle, font=f_sub, fill=INK_2)
    # legend (always present for 2 series): swatch + label in text ink
    lx = W - 16
    for i in (1, 0):
        tw_l = d.textlength(labels[i], font=f_small)
        lx -= tw_l
        d.text((lx, 22), labels[i], font=f_small, fill=INK)
        lx -= 16
        d.rounded_rectangle((lx, 23, lx + 10, 33), radius=3, fill=team_rgb[i])
        lx -= 14
    d.text((16, L.tactical[1] + 10), "LIVE TACTICAL VIEW", font=f_small, fill=INK)
    titles = ["SPACE CONTROL", "TEAM SPACING", "DISTANCE COVERED", "OPEN SPACE"]
    subs = ["frontcourt share", "convex hull area", "cumulative, all players", "offense gap to defender"]
    for bx, t, s_ in zip(tile_boxes, titles, subs):
        d.text((bx[0] + 10, bx[1] + 8), t, font=f_tile_t, fill=INK)
        d.text((bx[0] + 10, bx[1] + 22), s_, font=f_lbl, fill=INK_3)
    base = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # per-frame lookups
    tr_by_f = {f: g for f, g in tracks.groupby("frame")}
    fm = fm.set_index("frame")
    kp_lookup = dets[dets.cls == "person"].set_index(["frame", "det_id"]).kpts
    sm = Smoother(alpha=0.15, stale_after=int(rc.stale_after_frames))
    edge_ann = [sv.EdgeAnnotator(color=sv.Color(*team_rgb[i]), thickness=2, edges=[(a + 1, b + 1) for a, b in COCO_EDGES])
                for i in range(2)] + [sv.EdgeAnnotator(color=sv.Color(*ref_rgb), thickness=1,
                                                         edges=[(a + 1, b + 1) for a, b in COCO_EDGES])]
    ell_ann = sv.EllipseAnnotator(color=sv.ColorPalette([sv.Color(*c) for c in team_rgb] + [sv.Color(*ref_rgb)]),
                                  color_lookup=sv.ColorLookup.CLASS, thickness=2)
    # attacking pill follows the segmented possession (stable), not per-frame flicker
    poss_label = {}
    last_poss = None
    if "possession" in fm.columns:
        vv = fm[(fm.possession >= 0) & fm.valid & (fm.offense >= 0)]
        for pid, gq in vv.groupby("possession"):
            poss_label[pid] = (int(gq.offense.mode().iloc[0]), gq.attacking_side.mode().iloc[0])
    open_acc = {0: [], 1: []}

    with VideoWriter(out_path, info.fps, (W, Hh), cfg.output.h264) as writer:
        for fi, frame in iter_frames(video, start, end - start):
            img = base.copy()
            g = tr_by_f.get(fi, tracks.iloc[:0])
            row = fm.loc[fi] if fi in fm.index else None
            conf_ok = row is not None and float(row.get("calib_conf", 0)) >= thr
            valid = bool(row is not None and row.get("valid", False))

            # ---- broadcast panel
            bx0, by0, bx1, by1 = L.broadcast
            bf = cv2.resize(frame, (bx1 - bx0, by1 - by0), interpolation=cv2.INTER_AREA)
            show = g[g.team.isin([0, 1, 2]) & g.on_court]
            if len(show):
                xyxy = show[["x1", "y1", "x2", "y2"]].to_numpy(np.float32) * sx
                cls = show.team.to_numpy().astype(int)
                bf = ell_ann.annotate(bf, sv.Detections(xyxy=xyxy, class_id=cls))
                for team_i in (0, 1, 2):
                    sel = show[show.team == team_i]
                    kps, confs = [], []
                    for r in sel.itertuples():
                        k = kpts_from_row(kp_lookup.get((r.frame, r.det_id)))
                        if k is not None:
                            kps.append(k[:, :2] * sx)
                            confs.append(k[:, 2])
                    if kps:
                        xy = np.stack(kps).astype(np.float32)
                        cf = np.stack(confs).astype(np.float32)
                        xy[cf < 0.3] = 0  # supervision skips (0,0) vertices
                        bf = edge_ann[team_i].annotate(bf, sv.KeyPoints(xy=xy, keypoint_confidence=cf))
            img[by0:by1, bx0:bx1] = bf

            # ---- tactical panel: voronoi + dots
            ov = img.copy()
            pl = g[g.team.isin([0, 1]) & g.valid & g.x_ft.notna()]
            if conf_ok and len(pl) >= 2:
                pts = pl[["x_ft", "y_ft"]].to_numpy()
                cells = voronoi_cells(pts, (0, 0, court.length, court.width))
                for c, t in zip(cells, pl.team.to_numpy()):
                    if len(c) >= 3:
                        cv2.fillPoly(ov, [(canvas.px(c) * 16).round().astype(np.int32)], team_bgr[int(t)],
                                     cv2.LINE_AA, shift=4)
                        cv2.polylines(ov, [(canvas.px(c) * 16).round().astype(np.int32)], True, court_bg, 1,
                                      cv2.LINE_AA, shift=4)
                img = cv2.addWeighted(ov, 0.32, img, 0.68, 0)
                canvas.draw_lines(img)
            refs = g[(g.team == 2) & g.valid & g.x_ft.notna()]
            for r in refs.itertuples():
                p = canvas.px(np.clip([r.x_ft, r.y_ft], 0, [court.length, court.width]))[0].astype(int)
                cv2.circle(img, tuple(p), 4, ref_bgr, -1, cv2.LINE_AA)
            for r in pl.itertuples() if conf_ok else []:
                p = canvas.px(np.clip([r.x_ft, r.y_ft], 0, [court.length, court.width]))[0].astype(int)
                cv2.circle(img, tuple(p), 8, court_bg, -1, cv2.LINE_AA)       # 2 px surface ring
                cv2.circle(img, tuple(p), 6, team_bgr[int(r.team)], -1, cv2.LINE_AA)

            # ---- metrics for tiles
            if valid:
                for t in (0, 1):
                    sm.push(f"front{t}", row.get(f"front_control_pct_{t}"))
                    h = row.get(f"hull_ft2_{t}")
                    sm.push(f"hull{t}", h * FT2_TO_M2 if h is not None and np.isfinite(h) else None)
                off = int(row.get("offense", -1)) if np.isfinite(row.get("offense", np.nan)) else -1
                if off in (0, 1):
                    o = row.get("open_space_ft")
                    if o is not None and np.isfinite(o):
                        open_acc[off].append(o * FT_TO_M)
            else:
                for k in ("front0", "front1", "hull0", "hull1"):
                    sm.push(k, None)
            for t in (0, 1):
                dist = row.get(f"dist_m_{t}") if row is not None else None
                sm.v[f"dist{t}"] = float(dist) if dist is not None and np.isfinite(dist) else sm.v.get(f"dist{t}", 0.0)
                sm.age[f"dist{t}"] = 0
                sm.v[f"open{t}"] = float(np.mean(open_acc[t][-int(info.fps * 3):])) if open_acc[t] else None
                sm.age[f"open{t}"] = 0 if open_acc[t] else 10**6

            # ---- text layer
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(pil)
            for r in show[show.team.isin([0, 1])].itertuples():
                if np.isfinite(getattr(r, "speed_kmh", np.nan)):
                    cx = (r.x1 + r.x2) / 2 * sx + bx0
                    top = r.y1 * sx + by0 - 3
                    if top > by0 + 14:
                        _pill(d, (cx, top), f"{r.speed_kmh:.0f} km/h", f_small,
                              fill=team_rgb[int(r.team)] + (0,), ink=(255, 255, 255))
            if not conf_ok:
                _pill(d, (bx0 + 10, by0 + 10), "CALIBRATION LOW · ANALYTICS PAUSED", f_small,
                      fill=(127, 29, 29), anchor="lt")
            # attacking pill (majority over the last second)
            cur = poss_label.get(int(row["possession"])) if row is not None and "possession" in row else None
            last_poss = cur or last_poss  # hold through short gaps between possessions
            if last_poss is not None:
                off, side = last_poss
                arrow = "→" if side == "right" else "←"
                txt = f"{labels[off]} ATTACKING {arrow}" if side == "right" else f"{arrow} {labels[off]} ATTACKING"
                tw_ = d.textlength(txt, font=f_small) + 24
                _pill(d, (W - 16 - tw_, L.tactical[1] + 6), txt, f_small, fill=team_rgb[off], anchor="lt",
                      pad=(12, 4), ink=(255, 255, 255))
            # tiles
            specs = [("front", "{:.0f}%", None), ("hull", "{:.0f} m²", None), ("dist", "{:.0f} m", None),
                     ("open", "{:.1f} m", None)]
            for (key, fmt, _), bx in zip(specs, tile_boxes):
                va, vb = sm.get(f"{key}0"), sm.get(f"{key}1")
                stale = sm.stale(f"{key}0") and sm.stale(f"{key}1")
                ink = INK_3 if stale else INK
                ya = bx[1] + 44
                for i, v in enumerate((va, vb)):
                    xx = bx[0] + 10 + i * ((bx[2] - bx[0]) // 2)
                    d.rounded_rectangle((xx, ya + 5, xx + 8, ya + 13), radius=2, fill=team_rgb[i])
                    d.text((xx + 12, ya), "—" if v is None else fmt.format(v), font=f_val, fill=ink)
                    d.text((xx + 12, ya + 22), labels[i], font=f_lbl, fill=INK_3)
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            for (key, _, _), bx in zip(specs, tile_boxes):
                va, vb = sm.get(f"{key}0"), sm.get(f"{key}1")
                frac = None
                if va is not None and vb is not None and (va + vb) > 0:
                    frac = va / (va + vb)
                stale = sm.stale(f"{key}0") and sm.stale(f"{key}1")
                ca, cb = (team_bgr[0], team_bgr[1]) if not stale else (LINE, LINE)
                _split_bar(img, bx[0] + 10, bx[3] - 18, bx[2] - bx[0] - 20, 8, frac, ca, cb)
            writer.write(img)
            if (fi - start) % 200 == 0:
                log.info("render frame %d/%d", fi, end)
