"""
MODULE 4 — Oriented bounding box fitting.

L-shape fitting (Zhang et al., IV 2017 [70]): search over yaw hypotheses in
[0, 90) degrees, project the BEV points onto the two orthogonal edge
directions, and score each hypothesis with the *closeness* criterion
(points should hug two box edges, as vehicles seen by LiDAR form an "L").
The best yaw defines the rectangle; z extent comes from the point heights.

Corner alignment (OYSTER, Zhang et al., CVPR 2023 [67]) for MOVING objects:
because LiDAR only observes the sensor-facing surface, the geometric box
center is biased away from the ego vehicle. We therefore re-anchor the box
so that its corner CLOSEST to the ego vehicle coincides with the closest
observed point extent — applied later during refinement when the box size
is replaced by the track's median size (see label_refinement.py); at
fitting time the L-shape rectangle already tightly bounds the points.

Box format: (x, y, z, l, w, h, yaw) — center, dimensions, heading.
"""

import numpy as np


class Detection:
    """A fitted box plus its source segment; unit of tracking/refinement."""

    __slots__ = ("box", "segment", "frame_idx", "is_moving",
                 "clip_label", "clip_score", "track_id")

    def __init__(self, box, segment):
        self.box = box                  # (7,) in GLOBAL frame
        self.segment = segment          # clustering.Segment
        self.frame_idx = -1
        self.is_moving = False          # set by tracker (PP-score rule)
        self.clip_label = None
        self.clip_score = 0.0
        self.track_id = -1

    @property
    def num_points(self):
        return len(self.segment.points_global)


def fit_lshape_bev(points_xy, angle_res_deg=1.0, criterion="closeness"):
    """
    Search-based rectangle fitting (Zhang et al. 2017).

    Args:
        points_xy: (N,2) BEV points.
    Returns:
        (cx, cy, length, width, yaw) — length along the heading direction.
    """
    best_score, best_theta = -np.inf, 0.0
    thetas = np.deg2rad(np.arange(0.0, 90.0, angle_res_deg))
    for theta in thetas:
        e1 = np.array([np.cos(theta), np.sin(theta)])
        e2 = np.array([-np.sin(theta), np.cos(theta)])
        c1 = points_xy[:, 0] * e1[0] + points_xy[:, 1] * e1[1]
        c2 = points_xy[:, 0] * e2[0] + points_xy[:, 1] * e2[1]
        if criterion == "area":
            score = -(c1.max() - c1.min()) * (c2.max() - c2.min())
        elif criterion == "variance":
            score = _variance_criterion(c1, c2)
        else:
            score = _closeness_criterion(c1, c2)
        if score > best_score:
            best_score, best_theta = score, theta

    e1 = np.array([np.cos(best_theta), np.sin(best_theta)])
    e2 = np.array([-np.sin(best_theta), np.cos(best_theta)])
    c1 = points_xy[:, 0] * e1[0] + points_xy[:, 1] * e1[1]
    c2 = points_xy[:, 0] * e2[0] + points_xy[:, 1] * e2[1]
    d1 = c1.max() - c1.min()
    d2 = c2.max() - c2.min()
    center = 0.5 * (c1.max() + c1.min()) * e1 + 0.5 * (c2.max() + c2.min()) * e2

    # heading along the LONGER edge
    if d1 >= d2:
        length, width, yaw = d1, d2, best_theta
    else:
        length, width, yaw = d2, d1, best_theta + np.pi / 2.0
    return center[0], center[1], max(length, 1e-3), max(width, 1e-3), yaw


def _closeness_criterion(c1, c2, d0=0.01):
    """Reward points being close to one of the two nearest rectangle edges."""
    score = 0.0
    for c in (c1, c2):
        d_near = np.minimum(c.max() - c, c - c.min())
        d_near = np.maximum(d_near, d0)
        score += np.sum(1.0 / d_near)
    return score


def _variance_criterion(c1, c2):
    """Negative variance of distances to the nearest edge (Zhang et al.)."""
    score = 0.0
    for c in (c1, c2):
        d_near = np.minimum(c.max() - c, c - c.min())
        score -= np.var(d_near)
    return score


def closest_corner(box, anchor_xy):
    """Return BEV coordinates of the box corner closest to `anchor_xy`."""
    cx, cy, _, l, w, _, yaw = box
    cos, sin = np.cos(yaw), np.sin(yaw)
    corners = []
    for sx in (1, -1):
        for sy in (1, -1):
            dx, dy = sx * l / 2.0, sy * w / 2.0
            corners.append([cx + dx * cos - dy * sin, cy + dx * sin + dy * cos])
    corners = np.asarray(corners)
    d = np.linalg.norm(corners - np.asarray(anchor_xy)[None], axis=1)
    return corners[np.argmin(d)], int(np.argmin(d))


def align_box_to_corner(box, corner_target_xy, corner_index):
    """
    Translate `box` so that its `corner_index`-th corner lands on
    `corner_target_xy` (OYSTER-style corner alignment for moving objects).
    """
    box = box.copy()
    cur_corner, _ = _corner_by_index(box, corner_index)
    shift = np.asarray(corner_target_xy) - cur_corner
    box[0] += shift[0]
    box[1] += shift[1]
    return box


def _corner_by_index(box, idx):
    cx, cy, _, l, w, _, yaw = box
    cos, sin = np.cos(yaw), np.sin(yaw)
    signs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    sx, sy = signs[idx]
    dx, dy = sx * l / 2.0, sy * w / 2.0
    return np.array([cx + dx * cos - dy * sin, cy + dx * sin + dy * cos]), idx


class BoxFitter:
    """Fits an oriented 3D box to each cluster segment (GLOBAL frame)."""

    def __init__(self, cfg):
        self.angle_res = cfg.get("angle_resolution_deg", 1.0)
        self.criterion = cfg.get("criterion", "closeness")

    def fit(self, segment):
        """
        Args:
            segment: clustering.Segment
        Returns:
            Detection with box (x,y,z,l,w,h,yaw) in the GLOBAL frame,
            or None for degenerate segments.
        """
        pts = segment.points_global
        if len(pts) < 3:
            return None
        cx, cy, l, w, yaw = fit_lshape_bev(pts[:, :2], self.angle_res, self.criterion)
        z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
        h = max(z_max - z_min, 1e-3)
        box = np.array([cx, cy, 0.5 * (z_min + z_max), l, w, h, yaw], np.float64)
        return Detection(box, segment)
