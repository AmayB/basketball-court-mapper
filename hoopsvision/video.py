"""Streaming video I/O. Frames are yielded one at a time; nothing buffers the clip."""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    n_frames: int


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    info = VideoInfo(
        str(path),
        cap.get(cv2.CAP_PROP_FPS) or 30.0,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def iter_frames(path: str | Path, start: int = 0, max_frames: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx, n = start, 0
    try:
        while max_frames is None or n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
            n += 1
    finally:
        cap.release()


def frame_range(info: VideoInfo, cfg) -> tuple[int, int]:
    start = int(cfg.video.start_frame or 0)
    mx = cfg.video.max_frames
    end = info.n_frames if mx is None else min(info.n_frames, start + int(mx))
    return start, end


class VideoWriter:
    """Writes BGR frames. Prefers H.264 via the bundled ffmpeg (browser-friendly),
    falls back to OpenCV's mp4v encoder."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], h264: bool = True):
        self.path = str(path)
        self.size = size
        self.proc = None
        self.cv = None
        ffmpeg = _ffmpeg_exe() if h264 else None
        if ffmpeg:
            w, h = size
            cmd = [
                ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
                "-movflags", "+faststart", self.path,
            ]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size)
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        if self.cv is not None:
            self.cv.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - optional dependency
        log.info("imageio-ffmpeg unavailable; writing mp4v via OpenCV")
        return None
