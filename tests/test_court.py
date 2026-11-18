import numpy as np

from hoopsvision.config import load_config
from hoopsvision.court import CourtSpec, apply_h, image_to_court


def spec():
    return CourtSpec.from_cfg(load_config(None).court)


def test_dimensions_and_area():
    c = spec()
    assert c.area == 94 * 50
    assert np.allclose(c.hoop("right"), [94 - 5.25, 25])
    lc = c.lane_corners("left")
    assert np.allclose(lc["ft_far"], [19, 33])
    assert np.allclose(lc["base_near"], [0, 17])


def test_three_point_arc_is_at_radius():
    c = spec()
    arc = c._three_point("right")[2]
    d = np.hypot(*(arc - c.hoop("right")).T)
    assert np.allclose(d, 23.75)
    # arc bulges toward mid-court on the right end
    assert arc[:, 0].min() < c.hoop("right")[0]


def test_sample_points_inside_court():
    pts = spec().sample_points(1.0)
    assert len(pts) > 500
    assert spec().contains(pts, margin=1e-6).all()


def test_apply_h_identity_and_inverse():
    H = np.array([[2.0, 0.1, 5], [0.05, 1.5, -3], [1e-4, 2e-4, 1]])
    pts = np.random.default_rng(0).uniform(0, 50, (20, 2))
    assert np.allclose(apply_h(np.eye(3), pts), pts)
    assert np.allclose(image_to_court(H, apply_h(H, pts)), pts, atol=1e-8)
