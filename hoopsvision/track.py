"""Stage 2: ByteTrack association per shot segment.

- Occlusion: ByteTrack's lost_track_buffer keeps a track alive (Kalman-predicted)
  for N frames, so a player hidden behind a screen re-acquires the same ID.
  Low-confidence detections are used in ByteTrack's second association pass,
  which helps with partially occluded players.
- Cuts: a fresh tracker is created at each segment boundary and IDs are offset,
  so an ID never spans two camera shots (no smeared IDs across cuts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import supervision as sv


def _make_tracker(cfg, fps: float):
    from trackers import ByteTrackTracker

    t = cfg.tracking
    return ByteTrackTracker(
        lost_track_buffer=int(t.lost_track_buffer),
        frame_rate=float(fps),
        track_activation_threshold=float(t.track_activation_threshold),
        minimum_consecutive_frames=int(t.minimum_consecutive_frames),
        minimum_iou_threshold=float(t.minimum_iou_threshold),
        high_conf_det_threshold=float(t.high_conf_det_threshold),
    )


def run_tracking(dets: pd.DataFrame, segments: list[tuple[int, int]], cfg, fps: float) -> pd.DataFrame:
    persons = dets[dets.cls == "person"]
    by_frame = {f: g for f, g in persons.groupby("frame")}
    out = []
    id_offset = 0
    for seg_idx, (a, b) in enumerate(segments):
        tracker = _make_tracker(cfg, fps)
        max_id = -1
        for f in range(a, b):
            g = by_frame.get(f)
            if g is None or len(g) == 0:
                d = sv.Detections.empty()
                d.data["det_id"] = np.zeros(0, int)
            else:
                d = sv.Detections(
                    xyxy=g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
                    confidence=g.conf.to_numpy(np.float32),
                    class_id=np.zeros(len(g), int),
                    data={"det_id": g.det_id.to_numpy()},
                )
            res = tracker.update(d)
            if len(res) == 0:
                continue
            keep = res.tracker_id >= 0
            for xyxy, conf, tid, did in zip(
                res.xyxy[keep], res.confidence[keep], res.tracker_id[keep], res.data["det_id"][keep]
            ):
                max_id = max(max_id, int(tid))
                out.append((f, seg_idx, id_offset + int(tid), int(did), *map(float, xyxy), float(conf)))
        id_offset += max_id + 1
    df = pd.DataFrame(out, columns=["frame", "segment", "track_id", "det_id", "x1", "y1", "x2", "y2", "conf"])
    return filter_short_tracks(df, int(cfg.tracking.min_track_length))


def filter_short_tracks(df: pd.DataFrame, min_len: int) -> pd.DataFrame:
    if df.empty:
        return df
    n = df.groupby("track_id").frame.transform("size")
    return df[n >= min_len].reset_index(drop=True)
