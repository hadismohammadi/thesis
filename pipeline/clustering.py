"""
MODULE 3 — Spatio-temporal clustering.

Aggregates n consecutive non-ground frames into the reference frame t,
sub-samples each frame by 1/n while keeping ALL points whose PP-score
indicates motion, then runs HDBSCAN on per-point feature vectors

        f = [x, y, z, w_pp * pp_score, w_dt * delta_t]

The extra features let HDBSCAN (a) separate moving from static objects and
(b) separate two moving objects that occupy the same space at different
times (e.g. crossing pedestrians) — see ViLGOD Sec. 3.1.

ASSUMPTION (paper ambiguity): the paper does not state the feature scaling
of pp_score and delta_t relative to metric x,y,z. We expose both as config
weights (default 2.0), chosen so that a full PP-score swing (0->1) or one
aggregation window of time difference is comparable to ~2 m of spatial
separation. The request text said "keep all points with PP-score > 0";
the paper instead keeps points "where the PP-score indicates motion",
i.e. LOW persistence. We follow the paper: points with
pp < motion_keep_threshold bypass the 1/n sub-sampling.

Post-clustering filters (paper Appendix A):
    * remove segments with < 10 points
    * remove segments whose bottom is > 1 m above the ground plane O_g
    * remove segments with height < 0.5 m
"""

import numpy as np
import hdbscan

from .ground_removal import height_above_plane


class Segment:
    """A clustered point segment in the reference frame."""

    __slots__ = ("points_global", "points_sensor", "pp_scores", "ref_pose")

    def __init__(self, points_global, points_sensor, pp_scores, ref_pose):
        self.points_global = points_global      # (P,3) global frame
        self.points_sensor = points_sensor      # (P,3) reference SENSOR frame
        self.pp_scores = pp_scores              # (P,)
        self.ref_pose = ref_pose                # (4,4) sensor->global of ref frame


class SpatioTemporalClusterer:
    """HDBSCAN over temporally aggregated, motion-aware point features."""

    def __init__(self, cfg):
        self.cfg = cfg
        h = cfg["hdbscan"]
        self.clusterer_args = dict(
            min_cluster_size=h.get("min_cluster_size", 15),
            min_samples=h.get("min_samples", 15),
            cluster_selection_epsilon=float(h.get("cluster_selection_epsilon", 0.15)),
            core_dist_n_jobs=-1,
        )

    def cluster(self, frames_global, frames_pp, timestamps, ref_idx,
                ground_plane_sensor, ref_pose):
        """
        Args:
            frames_global: list of n (N,3) arrays (non-ground, global frame)
            frames_pp:     list of n (N,)  PP-score arrays
            timestamps:    list of n float seconds
            ref_idx:       index of the reference frame within the window
            ground_plane_sensor: (4,) plane O_g of the reference frame
                                 (SENSOR-frame coefficients)
            ref_pose:      (4,4) sensor->global pose of the reference frame
        Returns:
            list[Segment] — clustered object proposals (reference frame t).
        """
        n = len(frames_global)
        ratio = self.cfg.get("subsample_ratio", 1.0 / max(n, 1))
        motion_thr = self.cfg.get("motion_keep_threshold", 0.5)
        rng = np.random.default_rng(0)

        pts_list, pp_list, dt_list = [], [], []
        t_ref = timestamps[ref_idx]
        for j in range(n):
            pts, pp = frames_global[j], frames_pp[j]
            if len(pts) == 0:
                continue
            keep = rng.random(len(pts)) < ratio
            keep |= pp < motion_thr            # keep all moving points
            pts_list.append(pts[keep])
            pp_list.append(pp[keep])
            dt_list.append(np.full(keep.sum(), timestamps[j] - t_ref, np.float32))
        if not pts_list:
            return []

        P = np.vstack(pts_list).astype(np.float32)
        PP = np.concatenate(pp_list).astype(np.float32)
        DT = np.concatenate(dt_list).astype(np.float32)

        feats = np.column_stack([
            P,
            self.cfg.get("pp_feature_weight", 2.0) * PP,
            self.cfg.get("dt_feature_weight", 2.0) * DT,
        ])

        labels = hdbscan.HDBSCAN(**self.clusterer_args).fit_predict(feats)

        # --------------------------- post-clustering filters
        f = self.cfg["filters"]
        T_gs = np.linalg.inv(ref_pose)         # global -> sensor
        R_gs = T_gs[:3, :3].astype(np.float32)
        t_gs = T_gs[:3, 3].astype(np.float32)
        segments = []
        for lbl in np.unique(labels):
            if lbl < 0:
                continue
            mask = labels == lbl
            if mask.sum() < f.get("min_points", 10):
                continue
            seg_g = P[mask]
            seg_s = np.dot(seg_g, R_gs.T) + t_gs

            h_above = height_above_plane(seg_s, ground_plane_sensor)
            if h_above.min() > f.get("max_dist_above_ground", 1.0):
                continue                       # flying object (occlusion artefact)
            height = seg_s[:, 2].max() - seg_s[:, 2].min()
            if height < f.get("min_height", 0.5):
                continue                       # too flat (curb, noise)
            segments.append(Segment(seg_g, seg_s, PP[mask], ref_pose))
        return segments
