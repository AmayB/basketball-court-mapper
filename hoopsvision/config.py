"""Config loading. The YAML file overrides DEFAULTS key by key (deep merge)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "video": {"start_frame": 0, "max_frames": None},
    "models": {
        "detector_weights": "models/yolo26s.pt",
        "pose_weights": "models/yolo26n-pose.pt",
        "device": "auto",
        "imgsz": 1280,
        "person_conf": 0.25,
        "ball_conf": 0.15,
        "pose_conf": 0.15,
        "pose_match_iou": 0.45,
        "min_box_height_frac": 0.06,
    },
    "court": {
        "length_ft": 94.0,
        "width_ft": 50.0,
        "lane_width_ft": 16.0,
        "ft_line_from_baseline_ft": 19.0,
        "ft_circle_radius_ft": 6.0,
        "center_circle_radius_ft": 6.0,
        "hoop_from_baseline_ft": 5.25,
        "three_pt_radius_ft": 23.75,
        "three_pt_corner_offset_ft": 22.0,
        "on_court_margin_ft": 3.0,
    },
    "scenecut": {"hist_bins": [30, 32], "struct_threshold": 0.35, "threshold": 0.5,
                 "hist_struct_threshold": 0.5, "min_gap_frames": 10},
    "tracking": {
        "lost_track_buffer": 45,
        "track_activation_threshold": 0.25,
        "high_conf_det_threshold": 0.4,
        "minimum_consecutive_frames": 2,
        "minimum_iou_threshold": 0.1,
        "min_track_length": 8,
    },
    "teams": {
        "labels": ["HOME", "AWAY"],
        "home_jersey": "light",
        "torso_x": [0.28, 0.72],
        "torso_y": [0.18, 0.50],
        "court_color_dist": 14.0,
        "min_pixels": 25,
        "sample_every": 2,
        "min_votes": 3,
        "ref_min_share": 0.6,
    },
    "calibration": {
        "basket_side": "auto",
        "paint_lab": None,
        "paint_lab_tol": 22.0,
        "ignore_regions": [],
        "static_overlay_detect": True,
        "icp_iterations": 4,
        "icp_radius_px": [14, 9, 6, 4],
        "inlier_tol_px": 3.0,
        "template_spacing_ft": 1.0,
        "min_visible_template_pts": 60,
        "smoothing_alpha": 0.35,
        "propagation_decay": 0.97,
        "max_propagation_frames": 60,
        "paint_search_every": 3,
        "motion_method": "lk",
        "low_conf_threshold": 0.45,
        "manual_anchors": [],
    },
    "metrics": {
        "speed_window_s": 0.5,
        "max_speed_kmh": 36.0,
        "max_gap_frames": 6,
        "min_players_for_hull": 3,
    },
    "possession": {"min_duration_s": 2.5, "smoothing_s": 1.0, "ball_assign_dist_ft": 4.0},
    "render": {
        "width": 720,
        "height": 900,
        "title": "BASKETBALL AI ANALYTICS",
        "subtitle": "Player tracking · Court mapping · Tactical metrics",
        "team_colors": ["#3B82F6", "#D97706"],
        "referee_color": "#9CA3AF",
        "background": "#0B1220",
        "court_bg": "#0F1B33",
        "stale_after_frames": 15,
    },
    "output": {"debug_video": True, "h264": True},
}


class Cfg(dict):
    """dict with attribute access (cfg.models.imgsz) and nested Cfg values."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:  # pragma: no cover - programming error
            raise AttributeError(name) from e

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            return Cfg({k: Cfg.wrap(v) for k, v in obj.items()})
        return obj

    def section_hash(self, *sections: str) -> str:
        payload = json.dumps({s: self.get(s) for s in sections}, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Cfg:
    data: dict = {}
    if path is not None and Path(path).exists():
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    merged = _deep_merge(DEFAULTS, data)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return Cfg.wrap(merged)


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    b, g, r = hex_to_bgr(h)
    return (r, g, b)
