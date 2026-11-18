"""Per-stage wall-clock timing, logged and exported to JSON."""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("hoopsvision")


class StageTimer:
    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    @contextmanager
    def stage(self, name: str, n_frames: int | None = None):
        log.info("[%s] start", name)
        t0 = time.perf_counter()
        status = {"cached": False}
        try:
            yield status
        finally:
            dt = time.perf_counter() - t0
            rec = {"seconds": round(dt, 3), "cached": status["cached"]}
            if n_frames and not status["cached"]:
                rec["ms_per_frame"] = round(1000 * dt / max(n_frames, 1), 2)
            self.records[name] = rec
            log.info("[%s] %s in %.2fs%s", name, "cache hit" if status["cached"] else "done", dt,
                     f" ({rec['ms_per_frame']} ms/frame)" if "ms_per_frame" in rec else "")

    def dump(self, path: str | Path) -> None:
        """Write timings; for cache hits keep the last measured compute time so
        the file always shows what each stage really costs."""
        path = Path(path)
        old = json.loads(path.read_text()) if path.exists() else {}
        out = {}
        for k, rec in self.records.items():
            prev = old.get(k, {})
            if rec["cached"] and "ms_per_frame" in prev:
                rec = {**rec, "last_compute_seconds": prev.get("last_compute_seconds", prev["seconds"]),
                       "ms_per_frame": prev["ms_per_frame"]}
            out[k] = rec
        path.write_text(json.dumps(out, indent=2))
