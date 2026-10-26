"""
MODULE 1 — Ground removal.

Patchwork++ (Lee et al., IROS 2022) segments ground points on concentric
zone patches; we then fit a single global ground plane O_g with RANSAC on
the segmented ground points, exactly as in ViLGOD Sec. 3.1. The plane O_g
is later used by the clustering filters ("flying object" removal).

If pypatchworkpp is unavailable, a pure-RANSAC fallback (Open3D
segment_plane on the lowest height band) is used; it is noticeably worse
on slopes/curbs, so Patchwork++ is strongly recommended.
"""

import numpy as np

try:
    import pypatchworkpp
    _HAS_PWPP = True
except ImportError:
    _HAS_PWPP = False

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False


def ransac_plane(points, dist_thresh=0.15, iters=200, seed=0):
    """
    Fit a plane ax+by+cz+d=0 with RANSAC. Returns (plane(4,), inlier_mask).
    Pure-numpy implementation so we do not strictly depend on Open3D here.
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    if n < 3:
        return np.array([0.0, 0.0, 1.0, 0.0]), np.zeros(n, bool)
    best_inliers = None
    best_plane = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        d = -float(np.dot(normal, p0))
        dist = np.abs(points[:, 0] * normal[0] + points[:, 1] * normal[1] + points[:, 2] * normal[2] + d)
        inliers = dist < dist_thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers, best_plane = inliers, np.array([*normal, d])
    # least-squares refit on inliers
    if best_inliers is not None and best_inliers.sum() >= 3:
        P = points[best_inliers]
        centroid = P.mean(0)
        _, _, vt = np.linalg.svd(P - centroid, full_matrices=False)
        normal = vt[-1]
        if normal[2] < 0:  # make normal point up
            normal = -normal
        best_plane = np.array([*normal, -float(np.dot(normal, centroid))])
    return best_plane, best_inliers


class GroundRemover:
    """Removes ground points per frame; returns non-ground points + plane O_g."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.method = cfg.get("method", "patchworkpp")
        self.sensor_height = cfg.get("sensor_height", 1.84)
        # Translation of sensor relative to input coordinate frame (e.g. nuScenes ego frame)
        offset = cfg.get("sensor_offset", [0.0, 0.0, 0.0])
        self.sensor_offset = np.array(offset, dtype=np.float64)

        if self.method == "patchworkpp" and not _HAS_PWPP:
            print("[GroundRemover] pypatchworkpp not found -> RANSAC fallback "
                  "(install: github.com/url-kaist/patchwork-plusplus, `make pyinstall`)")
            self.method = "ransac"
        if self.method == "patchworkpp":
            params = pypatchworkpp.Parameters()
            params.sensor_height = float(self.sensor_height)
            params.th_dist = float(cfg.get("patchwork_th_dist", 0.20))
            params.enable_RNR = False  # Avoid warnings when intensity is omitted
            params.verbose = False
            self.pwpp = pypatchworkpp.patchworkpp(params)

    def remove(self, points_xyz):
        """
        Args:
            points_xyz: (N,3) float points in the frame defined by the dataset loader.
        Returns:
            nonground: (M,3) float32 non-ground points P_hat = P \\ G
            plane:     (4,)  ground plane O_g [a,b,c,d] with unit normal (up)
        """
        points_xyz = np.ascontiguousarray(points_xyz[:, :3], dtype=np.float64)
        n_pts = len(points_xyz)
        if n_pts == 0:
            return np.zeros((0, 3), dtype=np.float32), np.array([0.0, 0.0, 1.0, 0.0])

        # Center points at the LiDAR sensor origin for Patchwork++ (which expects sensor at [0,0,0])
        pts_sensor = points_xyz - self.sensor_offset

        if self.method == "patchworkpp":
            self.pwpp.estimateGround(pts_sensor)
            ground_idx = self.pwpp.getGroundIndices()
            ground_mask = np.zeros(n_pts, dtype=bool)
            ground_mask[ground_idx] = True

            ground_pts = points_xyz[ground_mask]
            if len(ground_pts) >= 3:
                plane, _ = ransac_plane(
                    ground_pts,
                    dist_thresh=self.cfg.get("ransac_dist_thresh", 0.15),
                    iters=self.cfg.get("ransac_iters", 200),
                )
            else:
                plane = np.array([0.0, 0.0, 1.0, 0.0])

            # 1. Residual ground filter: remove lingering road points very close to plane O_g
            residual_thresh = float(self.cfg.get("residual_thresh", 0.08))
            if residual_thresh > 0:
                dist_to_plane = height_above_plane(points_xyz, plane)
                is_near_ground = np.abs(dist_to_plane) < residual_thresh
                ground_mask |= is_near_ground

            # 2. Filter ego-vehicle self-reflections (within vehicle radius around sensor)
            min_r = float(self.cfg.get("min_sensor_radius", 0.0))
            if min_r > 0:
                dist_xy = np.linalg.norm(pts_sensor[:, :2], axis=1)
                is_ego_body = dist_xy < min_r
            else:
                is_ego_body = np.zeros(n_pts, dtype=bool)

            nonground_mask = (~ground_mask) & (~is_ego_body)
            nonground = points_xyz[nonground_mask]
            return nonground.astype(np.float32), plane

        # ------------------ RANSAC fallback ------------------
        # Restrict the plane hypothesis to a band around the expected ground
        # height in the sensor-relative coordinates.
        h = self.sensor_height
        z_rel = pts_sensor[:, 2]
        band = points_xyz[(z_rel > -h - 1.0) & (z_rel < -h + 1.0)]
        if len(band) < 50:
            band = points_xyz
        plane, _ = ransac_plane(
            band,
            dist_thresh=self.cfg.get("ransac_dist_thresh", 0.15),
            iters=self.cfg.get("ransac_iters", 200),
        )
        dist = height_above_plane(points_xyz, plane)
        nonground = points_xyz[dist > self.cfg.get("ransac_dist_thresh", 0.15)]
        return nonground.astype(np.float32), plane


def height_above_plane(points_xyz, plane):
    """Signed distance of points above the ground plane O_g."""
    return points_xyz[:, 0] * plane[0] + points_xyz[:, 1] * plane[1] + points_xyz[:, 2] * plane[2] + plane[3]
