"""Stage 1: detect people (players + referees), the ball, and pose keypoints.

A detection model supplies person/ball boxes (much better recall than pose models
on soft, low-resolution broadcast footage). A pose model, if configured, adds
17 COCO keypoints to person boxes matched by IoU (Hungarian assignment). People
are filtered by size here; the court-polygon filter (on-court vs bench/crowd)
happens after calibration, when foot points can be projected onto the court.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .video import iter_frames

log = logging.getLogger(__name__)

PERSON, BALL = 0, 32  # COCO class ids
N_KPT = 17


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between Nx4 and Mx4 xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_pose(person_boxes: np.ndarray, pose_boxes: np.ndarray, pose_kpts: np.ndarray, min_iou: float) -> np.ndarray:
    """Return (N,17,3) keypoints for each person box (NaN where unmatched)."""
    out = np.full((len(person_boxes), N_KPT, 3), np.nan, np.float32)
    iou = box_iou(person_boxes, pose_boxes)
    if iou.size == 0:
        return out
    r, c = linear_sum_assignment(-iou)
    for i, j in zip(r, c):
        if iou[i, j] >= min_iou:
            out[i] = pose_kpts[j]
    return out


class Detector:
    def __init__(self, cfg):
        from ultralytics import YOLO

        m = cfg.models
        self.cfg = m
        self.device = resolve_device(m.device)
        self.det = YOLO(_weights(m.detector_weights))
        self.pose = YOLO(_weights(m.pose_weights)) if m.pose_weights else None
        log.info("detector=%s pose=%s device=%s", m.detector_weights, m.pose_weights, self.device)

    def __call__(self, frames: list[np.ndarray]) -> list[dict]:
        m = self.cfg
        kw = dict(imgsz=m.imgsz, device=self.device, verbose=False)
        det_res = self.det(frames, conf=min(m.person_conf, m.ball_conf), classes=[PERSON, BALL], **kw)
        pose_res = self.pose(frames, conf=m.pose_conf, **kw) if self.pose else [None] * len(frames)
        out = []
        for frame, dr, pr in zip(frames, det_res, pose_res):
            xyxy = dr.boxes.xyxy.cpu().numpy()
            conf = dr.boxes.conf.cpu().numpy()
            cls = dr.boxes.cls.cpu().numpy().astype(int)
            min_h = m.min_box_height_frac * frame.shape[0]
            pmask = (cls == PERSON) & (conf >= m.person_conf) & ((xyxy[:, 3] - xyxy[:, 1]) >= min_h)
            bmask = (cls == BALL) & (conf >= m.ball_conf)
            pboxes = xyxy[pmask]
            if pr is not None and pr.keypoints is not None and len(pr.boxes):
                kp = np.concatenate(
                    [pr.keypoints.xy.cpu().numpy(), pr.keypoints.conf.cpu().numpy()[..., None]], -1
                )
                kpts = match_pose(pboxes, pr.boxes.xyxy.cpu().numpy(), kp, m.pose_match_iou)
            else:
                kpts = np.full((len(pboxes), N_KPT, 3), np.nan, np.float32)
            out.append(
                {"person_xyxy": pboxes, "person_conf": conf[pmask], "kpts": kpts,
                 "ball_xyxy": xyxy[bmask], "ball_conf": conf[bmask]}
            )
        return out


def _weights(p: str) -> str:
    # Accept a local path, or a bare model name that Ultralytics downloads.
    return str(p) if Path(p).exists() else Path(p).name


def run_detection(video: str, cfg, start: int, end: int, batch: int = 8) -> pd.DataFrame:
    """Stream the clip through the detector; return one row per detection."""
    det = Detector(cfg)
    rows: list[dict] = []
    buf: list[tuple[int, np.ndarray]] = []

    def flush():
        res = det([f for _, f in buf])
        for (fi, _), r in zip(buf, res):
            for k, (b, c) in enumerate(zip(r["person_xyxy"], r["person_conf"])):
                rows.append(_row(fi, k, "person", b, c, r["kpts"][k]))
            for k, (b, c) in enumerate(zip(r["ball_xyxy"], r["ball_conf"])):
                rows.append(_row(fi, k, "ball", b, c, None))
        buf.clear()

    for fi, frame in iter_frames(video, start, end - start):
        buf.append((fi, frame))
        if len(buf) == batch:
            flush()
        if fi % 100 == 0:
            log.info("detect frame %d/%d", fi, end)
    if buf:
        flush()
    df = pd.DataFrame(rows, columns=["frame", "det_id", "cls", "x1", "y1", "x2", "y2", "conf", "kpts"])
    return df


def _row(fi, k, cls, b, c, kp):
    return {
        "frame": int(fi), "det_id": int(k), "cls": cls,
        "x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]),
        "conf": float(c),
        "kpts": None if kp is None else np.nan_to_num(kp, nan=-1.0).astype(np.float32).ravel().tolist(),
    }


def kpts_from_row(v) -> np.ndarray | None:
    """Decode the stored keypoint list back to (17,3) (x, y, conf); None if absent."""
    if v is None or len(v) == 0:
        return None
    k = np.asarray(v, np.float32).reshape(N_KPT, 3)
    if (k[:, 2] < 0).all():
        return None
    return k
