"""Orchestrates the stages with per-stage caching and timing."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from .cache import StageCache, video_fingerprint
from .timing import StageTimer
from .video import frame_range, probe

log = logging.getLogger("hoopsvision")

STAGE_MODULES = {
    "detect": ("detect",),
    "scenecut": ("scenecut",),
    "track": ("track",),
    "calibrate": ("calibrate", "court"),
    "teams": ("teams", "project"),
    "metrics": ("metrics", "project", "court"),
}


def _code_hash(modules) -> str:
    h = hashlib.sha1()
    for m in modules:
        h.update((Path(__file__).parent / f"{m}.py").read_bytes())
    return h.hexdigest()[:10]


class Pipeline:
    def __init__(self, video: str, out: str, cfg, force: tuple[str, ...] = ()):
        self.video = str(video)
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "debug").mkdir(exist_ok=True)
        self.cfg = cfg
        self.info = probe(video)
        self.start, self.end = frame_range(self.info, cfg)
        self.n = self.end - self.start
        self.cache = StageCache(self.out / "cache", video_fingerprint(video))
        self.timer = StageTimer()
        self.force = set(force)
        self.rng = f"{self.start}:{self.end}"
        self.keys: dict[str, str] = {}

    # --- helpers -----------------------------------------------------------
    def _key(self, stage: str, sections: tuple[str, ...], upstream: tuple[str, ...]) -> str:
        """Cache key = clip range + config sections + upstream keys + this stage's
        source code, so editing a stage's module invalidates it (and downstream)."""
        parts = (self.rng, self.cfg.section_hash(*sections), _code_hash(STAGE_MODULES.get(stage, ()))) +             tuple(self.keys[u] for u in upstream)
        k = self.cache.key(stage, *parts)
        self.keys[stage] = k
        return k

    def _fresh(self, stage: str, key: str, *files: str) -> bool:
        return stage not in self.force and "all" not in self.force and self.cache.fresh(stage, key, *files)

    # --- stages ------------------------------------------------------------
    def detect(self):
        from .detect import run_detection

        k = self._key("detect", ("models",), ())
        with self.timer.stage("detect", self.n) as st:
            if self._fresh("detect", k, "detections.parquet"):
                st["cached"] = True
            else:
                df = run_detection(self.video, self.cfg, self.start, self.end)
                self.cache.save_df("detections.parquet", df)
                self.cache.commit("detect", k)
        return self.cache.load_df("detections.parquet")

    def scenecut(self):
        from .scenecut import run_scenecut

        k = self._key("scenecut", ("scenecut",), ())
        with self.timer.stage("scenecut", self.n) as st:
            if self._fresh("scenecut", k, "scenecut.json"):
                st["cached"] = True
            else:
                self.cache.save_json("scenecut.json", run_scenecut(self.video, self.cfg, self.start, self.end))
                self.cache.commit("scenecut", k)
        sc = self.cache.load_json("scenecut.json")
        sc["segments"] = [tuple(s) for s in sc["segments"]]
        log.info("scene cuts at frames %s -> %d segments", sc["cuts"], len(sc["segments"]))
        return sc

    def track(self, dets, sc):
        from .track import run_tracking

        k = self._key("track", ("tracking",), ("detect", "scenecut"))
        with self.timer.stage("track", self.n) as st:
            if self._fresh("track", k, "tracks_raw.parquet"):
                st["cached"] = True
            else:
                tr = run_tracking(dets, sc["segments"], self.cfg, self.info.fps)
                self.cache.save_df("tracks_raw.parquet", tr)
                self.cache.commit("track", k)
        tr = self.cache.load_df("tracks_raw.parquet")
        log.info("tracks: %d ids, %d rows", tr.track_id.nunique(), len(tr))
        return tr

    def calibrate(self, dets, sc):
        from .calibrate import run_calibration

        k = self._key("calibrate", ("calibration", "court"), ("detect", "scenecut"))
        with self.timer.stage("calibrate", self.n) as st:
            if self._fresh("calibrate", k, "calibration.npz", "calibration_meta.json"):
                st["cached"] = True
            else:
                cal = run_calibration(self.video, self.cfg, self.start, self.end, sc["segments"], dets)
                self.cache.save_npz("calibration.npz", H=cal["H"], conf=cal["conf"], overlay=cal["overlay"])
                self.cache.save_json("calibration_meta.json", cal["meta"])
                self.cache.commit("calibrate", k)
        z = self.cache.load_npz("calibration.npz")
        cal = {"H": z["H"], "conf": z["conf"], "overlay": z["overlay"], "meta": self.cache.load_json("calibration_meta.json")}
        self._overlay = cal["overlay"]
        thr = self.cfg.calibration.low_conf_threshold
        log.info("calibration: %.0f%% of frames >= confidence %.2f (mean %.2f)",
                 100 * (cal["conf"] >= thr).mean(), thr, cal["conf"].mean())
        return cal

    def project(self, tracks, dets, cal):
        from .project import project_tracks

        with self.timer.stage("project", self.n):
            pt = project_tracks(tracks, dets, cal["H"], cal["conf"], self.start, self.cfg)
        log.info("on-court tracks: %d of %d", pt[pt.on_court].track_id.nunique(), pt.track_id.nunique())
        return pt

    def teams(self, ptracks):
        from .teams import run_teams

        k = self._key("teams", ("teams", "calibration", "court"), ("track", "calibrate"))
        with self.timer.stage("teams", self.n) as st:
            if self._fresh("teams", k, "teams.json"):
                st["cached"] = True
            else:
                self.cache.save_json("teams.json", run_teams(self.video, ptracks, self.cfg, self.start, self.end, self._overlay))
                self.cache.commit("teams", k)
        res = self.cache.load_json("teams.json")
        res["track_team"] = {int(t): v for t, v in res["track_team"].items()}
        counts = np.bincount(list(res["track_team"].values()), minlength=3) if res["track_team"] else [0, 0, 0]
        log.info("teams: %d HOME / %d AWAY / %d referee-other tracks", *counts[:3])
        return res

    def metrics(self, ptracks, dets, cal):
        import json

        from .metrics import run_metrics

        k = self._key("metrics", ("metrics", "possession", "teams", "calibration", "court"), ("teams",))
        with self.timer.stage("metrics", self.n) as st:
            if self._fresh("metrics", k, "frame_metrics.parquet", "tracks_final.parquet", "possessions.json"):
                st["cached"] = True
            else:
                tracks, fm, summary = run_metrics(ptracks, dets, cal, self.cfg, self.info.fps, self.start, self.end)
                self.cache.save_df("tracks_final.parquet", tracks)
                self.cache.save_df("frame_metrics.parquet", fm)
                self.cache.save_json("possessions.json", summary)
                self.cache.commit("metrics", k)
        tracks = self.cache.load_df("tracks_final.parquet")
        fm = self.cache.load_df("frame_metrics.parquet")
        summary = self.cache.load_json("possessions.json")
        tracks.to_parquet(self.out / "tracks.parquet", index=False)
        (self.out / "possession_summary.json").write_text(json.dumps(
            {"video": Path(self.video).name, "fps": self.info.fps, "frames": [self.start, self.end],
             "team_labels": list(self.cfg.teams.labels),
             "valid_frame_pct": round(100 * float(fm.valid.mean()), 1),
             "possessions": summary}, indent=2))
        log.info("metrics: %.0f%% of frames valid for analytics, %d possessions",
                 100 * fm.valid.mean(), len(summary))
        return tracks, fm, summary

    def debug(self, cal, sc, ptracks):
        from .debug import plot_conf_timeline, write_debug_video

        with self.timer.stage("debug", self.n) as st:
            import pandas as pd

            dbg = self.out / "debug"
            plot_conf_timeline(cal, sc["cuts"], self.info.fps, self.start,
                               self.cfg.calibration.low_conf_threshold, dbg / "calibration_confidence.png")
            pd.DataFrame(cal["meta"]).assign(conf=cal["conf"]).to_csv(dbg / "calibration_timeline.csv", index=False)
            if self.cfg.output.debug_video:
                write_debug_video(self.video, dbg / "court_projection_debug.mp4", self.start, self.end, cal,
                                  self.cfg, ptracks, self.cfg.output.h264)
            else:
                st["cached"] = True

    def run(self, render: bool = True):
        dets = self.detect()
        sc = self.scenecut()
        tracks = self.track(dets, sc)
        cal = self.calibrate(dets, sc)
        ptracks = self.project(tracks, dets, cal)
        teams = self.teams(ptracks)
        ptracks["team"] = ptracks.track_id.map(teams["track_team"]).fillna(-1).astype(int)
        tracks, fm, summary = self.metrics(ptracks, dets, cal)
        self.debug(cal, sc, ptracks)
        if render:
            self.render(tracks, fm, dets)
        self.timer.dump(self.out / "stage_timings.json")
        return {"tracks": tracks, "frame_metrics": fm, "possessions": summary, "calibration": cal}

    def _load_for_render(self):
        """Everything the renderer needs, from cache only (no models)."""
        import pandas as pd

        need = ["tracks_final.parquet", "frame_metrics.parquet", "detections.parquet", "calibration.npz"]
        missing = [n for n in need if not self.cache.path(n).exists()]
        if missing:
            raise SystemExit(f"cache incomplete ({missing}); run `python -m hoopsvision run` first")
        return (self.cache.load_df("tracks_final.parquet"), self.cache.load_df("frame_metrics.parquet"),
                self.cache.load_df("detections.parquet"))

    def render(self, tracks=None, fm=None, dets=None):
        from .render import render_dashboard

        if tracks is None:
            tracks, fm, dets = self._load_for_render()
        z = self.cache.load_npz("calibration.npz")
        with self.timer.stage("render", self.n):
            render_dashboard(self.video, self.out / "annotated_dashboard.mp4", tracks, fm, dets, z["H"],
                             self.cfg, self.start, self.end)

    def render_only(self):
        self.render()
        self.timer.dump(self.out / "stage_timings_render.json")
