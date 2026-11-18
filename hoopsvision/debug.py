"""Debug outputs: projected-court-line overlay video and calibration timeline."""
from __future__ import annotations

import cv2
import numpy as np

from .court import CourtSpec, apply_h
from .video import VideoWriter, iter_frames


def draw_court_projection(frame: np.ndarray, H: np.ndarray, court: CourtSpec, color=(0, 255, 255), thickness=1):
    for pl in court.polylines():
        dense = _densify(pl, 0.5)
        uv = apply_h(H, dense)
        ok = np.isfinite(uv).all(1) & (np.abs(uv) < 1e5).all(1)
        # split at invalid points so we never draw across the horizon
        run = []
        for p, o in zip(uv, ok):
            if o:
                run.append(p)
            elif len(run) > 1:
                cv2.polylines(frame, [np.array(run, np.int32)], False, color, thickness, cv2.LINE_AA)
                run = []
            else:
                run = []
        if len(run) > 1:
            cv2.polylines(frame, [np.array(run, np.int32)], False, color, thickness, cv2.LINE_AA)
    return frame


def _densify(pl, step):
    out = [pl[0]]
    for a, b in zip(pl[:-1], pl[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step))
        out.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return np.array(out)


def conf_color(c: float, thr: float):
    return (60, 200, 60) if c >= thr else (0, 165, 255) if c >= thr * 0.5 else (60, 60, 230)


def write_debug_video(video, out_path, start, end, calib, cfg, tracks=None, h264=True):
    from .video import probe

    info = probe(video)
    court = CourtSpec.from_cfg(cfg.court)
    thr = cfg.calibration.low_conf_threshold
    meta = {m["frame"]: m for m in calib["meta"]}
    overlay = calib.get("overlay")
    by_frame = {f: g for f, g in tracks.groupby("frame")} if tracks is not None else {}
    with VideoWriter(out_path, info.fps, (info.width, info.height), h264) as w:
        for fi, frame in iter_frames(video, start, end - start):
            k = fi - start
            H = calib["H"][k]
            c = float(calib["conf"][k])
            m = meta.get(fi, {})
            if overlay is not None and overlay.any():
                cnts, _ = cv2.findContours(overlay.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, cnts, -1, (255, 0, 255), 1)
            if np.isfinite(H).all():
                draw_court_projection(frame, H, court, (0, 255, 255) if c >= thr else (0, 140, 255))
            for r in by_frame.get(fi, []).itertuples() if fi in by_frame else []:
                col = (255, 255, 255) if getattr(r, "on_court", True) else (120, 120, 120)
                cv2.circle(frame, (int((r.x1 + r.x2) / 2), int(r.y2)), 3, col, -1)
            col = conf_color(c, thr)
            cv2.rectangle(frame, (0, 0), (info.width, 22), (0, 0, 0), -1)
            txt = f"f{fi} conf {c:.2f} src {m.get('source', '?')} side {m.get('side') or '-'} " \
                  f"score {m.get('score', 0):.2f} vis {m.get('n_visible', 0)}"
            cv2.putText(frame, txt, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
            bw = int(info.width * min(c, 1.0))
            cv2.rectangle(frame, (0, 20), (bw, 22), col, -1)
            w.write(frame)


def plot_conf_timeline(calib, cuts, fps, start, thr, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = np.asarray(calib["conf"])
    t = (np.arange(len(c)) + start) / fps
    fig, ax = plt.subplots(figsize=(10, 2.6), dpi=110)
    ax.fill_between(t, c, color="#3B82F6", alpha=0.35, lw=0)
    ax.plot(t, c, color="#3B82F6", lw=1)
    ax.axhline(thr, color="#F59E0B", ls="--", lw=1, label=f"low-confidence threshold {thr}")
    for cut in cuts:
        ax.axvline(cut / fps, color="#9CA3AF", lw=0.8, ls=":")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("time (s)  ·  dotted = scene cut")
    ax.set_ylabel("calib. confidence")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
