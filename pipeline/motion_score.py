"""
MODULE 2 — Motion scoring with the Persistence Point score (PP-score).

Following MODEST (You et al., CVPR 2022 [63]) / Rote-DA [64], as adopted by
ViLGOD: for each 3D point we count its local neighbors (radius r) in the
N adjacent frames on each side (all frames aligned in a common world frame
using ego poses). The neighbor-count histogram over the 2N+1 window is
normalized into a distribution P and the PP-score is the normalized entropy

        pp(q) = H(P) / log(2N + 1)            in [0, 1].

Interpretation:
    HIGH score  -> neighbors evenly distributed over time -> PERSISTENT/static
    LOW  score  -> neighbors concentrated in few frames    -> EPHEMERAL/moving

Frame alignment:
    nuScenes — ego poses from the devkit (loader already provides
               sensor->global poses).
    KITTI    — oxts (GPS/IMU) poses chained with calib_imu_to_velo
               (handled in data/kitti_loader.py).
Both loaders hand this module points already expressed in the GLOBAL frame,
so the computation here is dataset-agnostic.
"""

import numpy as np
from scipy.spatial import cKDTree


class PPScoreComputer:
    """Computes per-point PP-scores for an aligned LiDAR sequence."""

    def __init__(self, cfg):
        self.radius = cfg.get("radius", 0.6)
        self.num_adjacent = cfg.get("num_adjacent", 3)
        self.max_points = cfg.get("max_points_kdtree", 200000)

    def compute_sequence(self, frames_global):
        """
        Args:
            frames_global: list of (N_t, 3) float32 arrays in the GLOBAL frame
                           (non-ground points only).
        Returns:
            list of (N_t,) float32 PP-scores in [0, 1], aligned with input.
        """
        T = len(frames_global)
        # Pre-build one KD-tree per frame (reused by all queries).
        trees = []
        for pts in frames_global:
            if len(pts) > self.max_points:
                # uniform subsample for tree construction only — counts are
                # rescaled below so the score stays comparable
                idx = np.random.default_rng(0).choice(len(pts), self.max_points, replace=False)
                trees.append((cKDTree(pts[idx]), len(pts) / self.max_points))
            else:
                trees.append((cKDTree(pts) if len(pts) else None, 1.0))

        scores = []
        for t, query in enumerate(frames_global):
            if len(query) == 0:
                scores.append(np.zeros(0, np.float32))
                continue
            lo = max(0, t - self.num_adjacent)
            hi = min(T - 1, t + self.num_adjacent)
            window = list(range(lo, hi + 1))
            counts = np.zeros((len(query), len(window)), np.float64)
            for j, fidx in enumerate(window):
                tree, scale = trees[fidx]
                if tree is None:
                    continue
                c = tree.query_ball_point(query, r=self.radius, return_length=True)
                counts[:, j] = np.asarray(c, np.float64) * scale

            total = counts.sum(1)
            valid = total > 0
            P = np.zeros_like(counts)
            P[valid] = counts[valid] / total[valid, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                logP = np.where(P > 0, np.log(P), 0.0)
            H = -(P * logP).sum(1)
            pp = np.zeros(len(query), np.float32)
            if len(window) > 1:
                pp[valid] = (H[valid] / np.log(len(window))).astype(np.float32)
            scores.append(np.clip(pp, 0.0, 1.0))
        return scores
