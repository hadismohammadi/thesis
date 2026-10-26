"""
KITTI sequential loader for ViLGOD.

KITTI's *object* benchmark only ships isolated frames, but ViLGOD requires
temporal context. We therefore load KITTI RAW drives (synced+rectified),
which provide:
    velodyne_points/data/*.bin   — HDL-64E scans (x,y,z,reflectance),
                                   ALREADY in the LiDAR frame
                                   (x=forward, y=left, z=up)
    oxts/data/*.txt              — GPS/IMU readings for ego-motion
    calib_imu_to_velo.txt        — IMU -> Velodyne extrinsics

Pose chain used for frame alignment:
    velo -> imu -> global(mercator)         T_w_velo = T_w_imu @ T_imu_velo

NOTE on coordinate frames (addressing the request's instruction to convert
from the camera frame): KITTI *velodyne* .bin files are natively in the
LiDAR frame, so no conversion is needed for the point clouds themselves.
The CAMERA frame (x=right, y=down, z=forward) only matters when we EXPORT
pseudo-labels in the official KITTI label format — that conversion
(lidar -> rect camera via Tr_velo_to_cam and R0_rect) happens in
pipeline/pseudo_label_writer.py.
"""

import glob
import os

import numpy as np

EARTH_RADIUS = 6378137.0


# ----------------------------------------------------------------------
# OXTS -> pose (port of the official KITTI raw devkit convertOxtsToPose)
# ----------------------------------------------------------------------
def _lat_lon_to_mercator(lat, lon, scale):
    mx = scale * lon * np.pi * EARTH_RADIUS / 180.0
    my = scale * EARTH_RADIUS * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
    return mx, my


def _rot_from_rpy(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def oxts_to_poses(oxts_files):
    """Convert a drive's oxts readings to 4x4 IMU->world poses."""
    poses = []
    scale = None
    origin = None
    for f in oxts_files:
        vals = np.loadtxt(f)
        lat, lon, alt, roll, pitch, yaw = vals[:6]
        if scale is None:
            scale = np.cos(lat * np.pi / 180.0)
        mx, my = _lat_lon_to_mercator(lat, lon, scale)
        t = np.array([mx, my, alt])
        R = _rot_from_rpy(roll, pitch, yaw)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        if origin is None:
            origin = np.linalg.inv(T)
        poses.append(origin @ T)  # express relative to first frame
    return poses


def _read_calib_file(path):
    """Read a KITTI calibration text file into a dict of arrays."""
    out = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            try:
                out[k.strip()] = np.array([float(x) for x in v.split()])
            except ValueError:
                pass
    return out


def load_imu_to_velo(drive_date_dir):
    """4x4 IMU -> Velodyne transform from calib_imu_to_velo.txt."""
    c = _read_calib_file(os.path.join(drive_date_dir, "calib_imu_to_velo.txt"))
    T = np.eye(4)
    T[:3, :3] = c["R"].reshape(3, 3)
    T[:3, 3] = c["T"]
    return T


def load_velo_to_cam(drive_date_dir_or_calib):
    """
    4x4 Velodyne -> rectified cam0 transform.
    Accepts either a raw-drive date dir (calib_velo_to_cam.txt +
    calib_cam_to_cam.txt) or an object-benchmark calib file path.
    """
    if os.path.isdir(drive_date_dir_or_calib):
        vc = _read_calib_file(os.path.join(drive_date_dir_or_calib, "calib_velo_to_cam.txt"))
        cc = _read_calib_file(os.path.join(drive_date_dir_or_calib, "calib_cam_to_cam.txt"))
        T_vc = np.eye(4)
        T_vc[:3, :3] = vc["R"].reshape(3, 3)
        T_vc[:3, 3] = vc["T"]
        R0 = np.eye(4)
        R0[:3, :3] = cc["R_rect_00"].reshape(3, 3)
        return R0 @ T_vc
    # object benchmark calib file
    c = _read_calib_file(drive_date_dir_or_calib)
    T_vc = np.eye(4)
    T_vc[:3, :] = c["Tr_velo_to_cam"].reshape(3, 4)
    R0 = np.eye(4)
    R0[:3, :3] = c["R0_rect"].reshape(3, 3)
    return R0 @ T_vc


class KittiSequenceLoader:
    """Iterates over KITTI raw drives, yielding aligned LiDAR sequences."""

    def __init__(self, cfg):
        d = cfg["dataset"]
        self.raw_root = d["raw_root"]
        self.max_range = d.get("max_range", 120.0)
        drives = d.get("drives", "all")
        if drives == "all":
            self.drives = sorted(glob.glob(os.path.join(self.raw_root, "*", "*_sync")))
        else:
            self.drives = []
            for name in drives:
                date = "_".join(name.split("_")[:3])
                self.drives.append(os.path.join(self.raw_root, date, name))

    def __len__(self):
        return len(self.drives)

    def __iter__(self):
        for drive in self.drives:
            yield self.load_drive(drive)

    # ------------------------------------------------------------------
    def load_drive(self, drive_dir):
        """Load one raw drive as a sequence dict consumed by run_pipeline."""
        date_dir = os.path.dirname(drive_dir)
        velo_files = sorted(glob.glob(os.path.join(drive_dir, "velodyne_points", "data", "*.bin")))
        oxts_files = sorted(glob.glob(os.path.join(drive_dir, "oxts", "data", "*.txt")))
        ts_file = os.path.join(drive_dir, "velodyne_points", "timestamps.txt")
        timestamps = self._read_timestamps(ts_file, len(velo_files))

        imu_poses = oxts_to_poses(oxts_files)               # imu -> world
        T_imu_velo = load_imu_to_velo(date_dir)             # imu -> velo
        T_velo_imu = np.linalg.inv(T_imu_velo)              # velo -> imu

        frames = []
        for i, vf in enumerate(velo_files):
            pts = np.fromfile(vf, dtype=np.float32).reshape(-1, 4)
            r = np.linalg.norm(pts[:, :2], axis=1)
            pts = pts[(r < self.max_range) & (r > 2.5)]     # drop ego returns
            pose = imu_poses[i] @ T_velo_imu                # velo -> world
            frames.append({
                "points": pts,                              # LiDAR frame
                "pose": pose,
                "token": f"{os.path.basename(drive_dir)}_{i:010d}",
                "frame_index": i,
                "timestamp": timestamps[i],
                "is_keyframe": True,
            })
        return {
            "name": os.path.basename(drive_dir),
            "frames": frames,
            "dataset": "kitti",
            "date_dir": date_dir,
        }

    @staticmethod
    def _read_timestamps(path, n):
        """Parse KITTI timestamps.txt to float seconds; fall back to 10 Hz."""
        if not os.path.exists(path):
            return [i * 0.1 for i in range(n)]
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # format: 2011-09-26 13:02:25.964389445
                hms = line.split(" ")[1]
                h, m, s = hms.split(":")
                out.append(int(h) * 3600 + int(m) * 60 + float(s))
        return out if len(out) == n else [i * 0.1 for i in range(n)]
