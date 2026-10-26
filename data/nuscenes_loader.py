"""
nuScenes sequential loader for ViLGOD.

Wraps nuscenes-devkit to yield full scenes (sequences) of LiDAR sweeps with
poses. ViLGOD needs *sequences* with accurate frame-to-frame alignment;
nuScenes provides this via per-sweep ego poses and the LIDAR_TOP extrinsics.

Frame convention (returned by this loader):
    points : (N, 4) float32 in the SENSOR (LIDAR_TOP) frame, columns x,y,z,intensity
             nuScenes LiDAR frame: x=right, y=forward, z=up — note this differs
             from the ego frame (x=forward). We rotate sweeps into the
             EGO-vehicle convention (x=forward, y=left, z=up) so the entire
             pipeline (and KITTI after conversion) shares one convention.
    pose   : (4,4) float64 transform SENSOR(ego-aligned) -> GLOBAL frame.
    token  : sample_data token (used as frame id when exporting labels)
    is_keyframe : bool — pseudo-labels are only exported on keyframes.

Install: pip install nuscenes-devkit
"""

import os

import numpy as np

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    _HAS_NUSC = True
except ImportError:  # allow importing the module without the devkit installed
    _HAS_NUSC = False


# nuScenes official train/val scene split sizes: 700 train / 150 val.
def _scene_split(nusc, split):
    """Return scene records belonging to the requested split."""
    from nuscenes.utils.splits import create_splits_scenes
    splits = create_splits_scenes()
    names = set(splits[split])
    return [s for s in nusc.scene if s["name"] in names]


class NuScenesSequenceLoader:
    """Iterates over nuScenes scenes, yielding aligned LiDAR sequences."""

    def __init__(self, cfg):
        if not _HAS_NUSC:
            raise ImportError("nuscenes-devkit not installed: pip install nuscenes-devkit")
        d = cfg["dataset"]
        self.nusc = NuScenes(version=d["version"], dataroot=d["root"], verbose=False)
        self.use_sweeps = d.get("use_sweeps", True)
        self.max_range = d.get("max_range", 70.0)
        if d["version"] == "v1.0-mini":
            self.scenes = self.nusc.scene
        else:
            self.scenes = _scene_split(self.nusc, d.get("split", "train"))

    def __len__(self):
        return len(self.scenes)

    def __iter__(self):
        for scene in self.scenes:
            yield self.load_scene(scene)

    # ------------------------------------------------------------------
    def _sweep_tokens(self, scene):
        """Collect all LIDAR_TOP sample_data tokens of a scene in time order."""
        first_sample = self.nusc.get("sample", scene["first_sample_token"])
        sd_token = first_sample["data"]["LIDAR_TOP"]
        # walk back to the very first sweep of the scene
        sd = self.nusc.get("sample_data", sd_token)
        while sd["prev"]:
            sd = self.nusc.get("sample_data", sd["prev"])
        tokens = []
        while True:
            tokens.append(sd["token"])
            if not sd["next"]:
                break
            sd = self.nusc.get("sample_data", sd["next"])
        return tokens

    def _load_frame(self, sd_token):
        """Load one sweep: points in ego-aligned sensor frame + global pose."""
        sd = self.nusc.get("sample_data", sd_token)
        pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, sd["filename"]))
        pts = pc.points.T[:, :4].astype(np.float32)  # (N,4) x,y,z,intensity (sensor frame)

        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep = self.nusc.get("ego_pose", sd["ego_pose_token"])

        # sensor -> ego
        T_se = transform_matrix(cs["translation"], Quaternion(cs["rotation"]))
        # ego -> global
        T_eg = transform_matrix(ep["translation"], Quaternion(ep["rotation"]))

        # Move points into the EGO frame (x=forward, y=left, z=up)
        R_se = T_se[:3, :3].astype(np.float32)
        t_se = T_se[:3, 3].astype(np.float32)
        pts_ego = np.dot(pts[:, :3], R_se.T) + t_se
        pts = np.hstack([pts_ego, pts[:, 3:4]])

        # range filter
        r = np.linalg.norm(pts[:, :2], axis=1)
        pts = pts[r < self.max_range]

        return {
            "points": pts,                                  # ego frame
            "pose": T_eg,                                   # ego -> global
            "token": sd_token,
            "timestamp": sd["timestamp"] * 1e-6,            # seconds
            "is_keyframe": sd["is_key_frame"],
        }

    def load_scene(self, scene):
        """Load a full scene as a sequence dict consumed by run_pipeline."""
        tokens = self._sweep_tokens(scene)
        if not self.use_sweeps:
            tokens = [t for t in tokens
                      if self.nusc.get("sample_data", t)["is_key_frame"]]
        frames = [self._load_frame(t) for t in tokens]
        return {"name": scene["name"], "frames": frames, "dataset": "nuscenes"}
