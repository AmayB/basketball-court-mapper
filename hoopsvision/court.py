"""Court geometry in feet. Pure functions, no I/O.

Coordinate frame: origin at the corner where the left baseline meets the near
(camera-side) sideline. x runs along the length (0..94), y across the width
(0..50, y=0 is the near sideline, at the bottom of a broadcast frame).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FT_TO_M = 0.3048


@dataclass(frozen=True)
class CourtSpec:
    length: float = 94.0
    width: float = 50.0
    lane_width: float = 16.0
    ft_line: float = 19.0
    ft_circle_r: float = 6.0
    center_circle_r: float = 6.0
    hoop_x: float = 5.25
    three_r: float = 23.75
    three_corner: float = 22.0

    @classmethod
    def from_cfg(cls, c) -> "CourtSpec":
        return cls(
            length=c["length_ft"],
            width=c["width_ft"],
            lane_width=c["lane_width_ft"],
            ft_line=c["ft_line_from_baseline_ft"],
            ft_circle_r=c["ft_circle_radius_ft"],
            center_circle_r=c["center_circle_radius_ft"],
            hoop_x=c["hoop_from_baseline_ft"],
            three_r=c["three_pt_radius_ft"],
            three_corner=c["three_pt_corner_offset_ft"],
        )

    @property
    def area(self) -> float:
        return self.length * self.width

    @property
    def mid_y(self) -> float:
        return self.width / 2

    def baseline_x(self, side: str) -> float:
        return 0.0 if side == "left" else self.length

    def direction(self, side: str) -> float:
        """+1 when moving from the given baseline toward mid-court increases x."""
        return 1.0 if side == "left" else -1.0

    def hoop(self, side: str) -> np.ndarray:
        return np.array([self.baseline_x(side) + self.direction(side) * self.hoop_x, self.mid_y])

    def lane_corners(self, side: str) -> dict[str, np.ndarray]:
        """Lane (paint) corners keyed by role. 'far' = larger y (far sideline)."""
        b, d = self.baseline_x(side), self.direction(side)
        ft = b + d * self.ft_line
        y0, y1 = self.mid_y - self.lane_width / 2, self.mid_y + self.lane_width / 2
        return {
            "base_near": np.array([b, y0]),
            "base_far": np.array([b, y1]),
            "ft_near": np.array([ft, y0]),
            "ft_far": np.array([ft, y1]),
        }

    def half_rect(self, side: str) -> tuple[float, float, float, float]:
        half = self.length / 2
        return (0.0, 0.0, half, self.width) if side == "left" else (half, 0.0, self.length, self.width)

    def polylines(self) -> list[np.ndarray]:
        """All painted court lines as polylines (Nx2, feet)."""
        L, W = self.length, self.width
        lines: list[np.ndarray] = [
            np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], float),
            np.array([[L / 2, 0], [L / 2, W]], float),
            _circle((L / 2, W / 2), self.center_circle_r),
        ]
        for side in ("left", "right"):
            b, d = self.baseline_x(side), self.direction(side)
            c = self.lane_corners(side)
            lines.append(np.array([c["base_near"], c["ft_near"], c["ft_far"], c["base_far"]]))
            lines.append(_circle((b + d * self.ft_line, W / 2), self.ft_circle_r))
            lines.extend(self._three_point(side))
        return lines

    def _three_point(self, side: str) -> list[np.ndarray]:
        b, d = self.baseline_x(side), self.direction(side)
        hx, hy = self.hoop(side)
        # where the arc meets the straight corner segment
        dx = np.sqrt(max(self.three_r**2 - self.three_corner**2, 0.0))
        join_x = hx + d * dx
        corner_a = np.array([[b, hy - self.three_corner], [join_x, hy - self.three_corner]])
        corner_b = np.array([[b, hy + self.three_corner], [join_x, hy + self.three_corner]])
        phi = np.arcsin(self.three_corner / self.three_r)
        ang = np.linspace(-phi, phi, 60)
        arc = np.stack([hx + d * self.three_r * np.cos(ang), hy + self.three_r * np.sin(ang)], 1)
        return [corner_a, corner_b, arc]

    def sample_points(self, spacing: float = 1.0) -> np.ndarray:
        """Evenly spaced points along every court line (template for line fitting)."""
        pts = [_resample(pl, spacing) for pl in self.polylines()]
        return np.concatenate(pts, 0)

    def contains(self, xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
        xy = np.atleast_2d(xy)
        return (
            (xy[:, 0] >= -margin)
            & (xy[:, 0] <= self.length + margin)
            & (xy[:, 1] >= -margin)
            & (xy[:, 1] <= self.width + margin)
        )


def _circle(center, r, n=72) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n + 1)
    return np.stack([center[0] + r * np.cos(t), center[1] + r * np.sin(t)], 1)


def _resample(pl: np.ndarray, spacing: float) -> np.ndarray:
    seg = np.diff(pl, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0], np.cumsum(seglen)])
    total = cum[-1]
    if total == 0:
        return pl[:1]
    s = np.arange(0, total + 1e-9, spacing)
    x = np.interp(s, cum, pl[:, 0])
    y = np.interp(s, cum, pl[:, 1])
    return np.stack([x, y], 1)


# --- homography helpers (court <-> image) -----------------------------------

def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to Nx2 points."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ H.T
    w = hom[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = hom[:, :2] / w
    out[np.abs(w[:, 0]) < 1e-12] = np.nan
    return out


def image_to_court(H_court_to_img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    return apply_h(np.linalg.inv(H_court_to_img), uv)
