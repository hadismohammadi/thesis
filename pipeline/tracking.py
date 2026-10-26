"""
MODULE 5 — Multi-target tracking + motion classification.

Greedy assignment (ViLGOD Appendix A):
    * match track predictions to detections within a 1 m Euclidean radius
      (BEV center distance), greedily by increasing distance;
    * if a track finds no match, RELAX the radius to 5 m and accept a match
      only if the cluster point counts differ by < 30 % — this recovers very
      fast movers (highway vehicles) without causing identity switches;
    * unmatched detections start new tracks; tracks die after `max_age`
      consecutive unmatched frames.

Per-BOX motion status (paper Sec. 3.1):
    a box is STATIC if at least alpha=20 % of its segment's points have
    PP-score > delta=0.7 (high persistence), and MOVING otherwise.

Per-TRACK static rule:
    a track is static iff (1) every box overlaps the track's largest box
    (BEV IoU > 0) AND (2) no box was classified as moving — i.e. the track
    shows no smooth linear displacement either.
"""

import numpy as np


def bev_iou(box_a, box_b):
    """Axis-aligned BEV IoU approximation via the boxes' AABBs.

    Sufficient for the static-track overlap test (we only check IoU>0 on
    boxes of the SAME object at different times, where orientation is
    nearly identical)."""
    def aabb(b):
        cx, cy, _, l, w, _, yaw = b
        ext = 0.5 * (abs(l * np.cos(yaw)) + abs(w * np.sin(yaw))), \
              0.5 * (abs(l * np.sin(yaw)) + abs(w * np.cos(yaw)))
        return cx - ext[0], cy - ext[1], cx + ext[0], cy + ext[1]

    ax0, ay0, ax1, ay1 = aabb(box_a)
    bx0, by0, bx1, by1 = aabb(box_b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


class Track:
    """A tracked object: ordered list of detections over time."""

    _next_id = 0

    def __init__(self, det):
        self.id = Track._next_id
        Track._next_id += 1
        self.detections = [det]
        self.misses = 0
        self.is_static = None          # decided after the sequence ends
        det.track_id = self.id

    @property
    def last(self):
        return self.detections[-1]

    def predict_center(self):
        """Constant-velocity BEV prediction of the next center."""
        if len(self.detections) < 2:
            return self.last.box[:2].copy()
        c1 = self.detections[-1].box[:2]
        c0 = self.detections[-2].box[:2]
        dt = self.detections[-1].frame_idx - self.detections[-2].frame_idx
        v = (c1 - c0) / max(dt, 1)
        return c1 + v * (self.misses + 1)

    def add(self, det):
        det.track_id = self.id
        self.detections.append(det)
        self.misses = 0

    # ------------------------------------------------------------------
    def classify_static(self):
        """Apply the per-track static rule (see module docstring)."""
        if any(d.is_moving for d in self.detections):
            self.is_static = False
            return
        largest = max(self.detections, key=lambda d: d.box[3] * d.box[4])
        overlaps = all(bev_iou(d.box, largest.box) > 0.0 for d in self.detections)
        # smooth linear motion check: total displacement of centers
        centers = np.stack([d.box[:2] for d in self.detections])
        displacement = np.linalg.norm(centers[-1] - centers[0])
        smooth_motion = displacement > 2.0 and self._is_smooth(centers)
        self.is_static = overlaps and not smooth_motion

    @staticmethod
    def _is_smooth(centers, tol=0.7):
        """Heuristic: centers lie close to the straight start->end line."""
        if len(centers) < 3:
            return True
        d = centers[-1] - centers[0]
        n = np.linalg.norm(d)
        if n < 1e-6:
            return False
        d = d / n
        rel = centers - centers[0]
        perp = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
        return perp.mean() < tol


class GreedyTracker:
    """Greedy IoU/distance multi-target tracker over a full sequence."""

    def __init__(self, cfg):
        self.r = cfg.get("match_radius", 1.0)
        self.r_relaxed = cfg.get("relaxed_radius", 5.0)
        self.max_pt_diff = cfg.get("relaxed_max_point_diff", 0.30)
        self.max_age = cfg.get("max_age", 3)
        self.alpha = cfg.get("static_alpha", 0.20)
        self.delta = cfg.get("static_delta", 0.70)

    def run(self, detections, num_frames):
        """
        Args:
            detections: dict frame_idx -> list[Detection]
        Returns:
            list[Track] (each with is_static decided).
        """
        # per-box motion classification from PP-scores
        for dets in detections.values():
            for d in dets:
                frac_persistent = float(np.mean(d.segment.pp_scores > self.delta))
                d.is_moving = frac_persistent < self.alpha

        active, finished = [], []
        for t in range(num_frames):
            dets = detections.get(t, [])
            unmatched_dets = set(range(len(dets)))
            unmatched_tracks = set(range(len(active)))

            # ---- strict pass (1 m radius), greedy by distance
            pairs = []
            for ti in unmatched_tracks:
                pred = active[ti].predict_center()
                for di in unmatched_dets:
                    dist = np.linalg.norm(pred - dets[di].box[:2])
                    if dist <= self.r:
                        pairs.append((dist, ti, di))
            for dist, ti, di in sorted(pairs):
                if ti in unmatched_tracks and di in unmatched_dets:
                    active[ti].add(dets[di])
                    unmatched_tracks.discard(ti)
                    unmatched_dets.discard(di)

            # ---- relaxed pass (5 m + <30% point-count difference)
            pairs = []
            for ti in unmatched_tracks:
                pred = active[ti].predict_center()
                n_t = active[ti].last.num_points
                for di in unmatched_dets:
                    n_d = dets[di].num_points
                    pt_diff = abs(n_t - n_d) / max(n_t, n_d, 1)
                    dist = np.linalg.norm(pred - dets[di].box[:2])
                    if dist <= self.r_relaxed and pt_diff < self.max_pt_diff:
                        pairs.append((dist, ti, di))
            for dist, ti, di in sorted(pairs):
                if ti in unmatched_tracks and di in unmatched_dets:
                    active[ti].add(dets[di])
                    unmatched_tracks.discard(ti)
                    unmatched_dets.discard(di)

            # ---- bookkeeping
            for ti in unmatched_tracks:
                active[ti].misses += 1
            survivors = []
            for tr in active:
                if tr.misses > self.max_age:
                    finished.append(tr)
                else:
                    survivors.append(tr)
            active = survivors
            for di in unmatched_dets:
                active.append(Track(dets[di]))

        finished.extend(active)
        for tr in finished:
            tr.classify_static()
        return finished
