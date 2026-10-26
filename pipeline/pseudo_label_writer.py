"""
MODULE 9 (part 1) — Pseudo-label export for OpenPCDet self-training.

KITTI:
    one .txt per frame in the official label format:
        <class> trunc occl alpha x1 y1 x2 y2 h w l x y z ry [score]
    Location (x,y,z) is the BOTTOM-CENTER in the RECTIFIED CAMERA frame
    (x=right, y=down, z=forward); ry is the rotation about the camera
    y-axis. We convert from our LiDAR-frame boxes via R0_rect @ Tr_velo_to_cam.
    2D box fields are zeroed (we are LiDAR-only) — OpenPCDet's KITTI loader
    ignores them when `FOV_POINTS_ONLY=False` and 2D AP is not evaluated.

nuScenes:
    a JSON file (per sequence) + a pickle of OpenPCDet-style "infos"
    entries containing gt_boxes (x,y,z,l,w,h,yaw in the LiDAR/ego frame of
    each keyframe) and gt_names. self_training/train_centerpoint.py turns
    these into the nuscenes_infos_*.pkl that NuScenesDataset consumes.
"""

import json
import os
import pickle

import numpy as np

from data.kitti_loader import load_velo_to_cam


def global_to_frame(box_global, pose):
    """Transform a (7,) global-frame box into the frame's SENSOR frame."""
    T = np.linalg.inv(pose)
    b = box_global.copy()
    center = T @ np.array([b[0], b[1], b[2], 1.0])
    b[:3] = center[:3]
    # rotate the heading: yaw is about +z; extract the z-rotation of T
    yaw_T = np.arctan2(T[1, 0], T[0, 0])
    b[6] = b[6] + yaw_T
    return b


class PseudoLabelWriter:
    """Writes refined detections as dataset-specific pseudo-label files."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.fmt = cfg["pseudo_labels"]["format"]
        self.min_score = cfg["pseudo_labels"].get("min_score", 0.0)
        self.out_dir = os.path.join(cfg["output_dir"], "pseudo_labels")
        os.makedirs(self.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def write_sequence(self, seq, final_dets):
        """Dispatch to the dataset backend; returns the per-frame results."""
        frames = seq["frames"]
        results = []
        for t, fr in enumerate(frames):
            dets = [d for d in final_dets.get(t, []) if d["score"] >= self.min_score]
            local = []
            for d in dets:
                local.append({
                    "box": global_to_frame(d["box"], fr["pose"]),
                    "label": d["label"],
                    "score": d["score"],
                    "track_id": d["track_id"],
                })
            results.append({"token": fr["token"],
                            "is_keyframe": fr.get("is_keyframe", True),
                            "detections": local})
        if self.fmt == "kitti":
            self._write_kitti(seq, results)
        else:
            self._write_nuscenes(seq, results)
        return results

    # ------------------------------------------------------------------
    def _write_kitti(self, seq, results):
        """One KITTI-format .txt per frame (camera-frame boxes)."""
        T_velo_cam = load_velo_to_cam(seq["date_dir"])      # 4x4 lidar->rect cam
        yaw_T = np.arctan2(T_velo_cam[1, 0], T_velo_cam[0, 0])
        label_dir = os.path.join(self.out_dir, seq["name"])
        os.makedirs(label_dir, exist_ok=True)

        for r in results:
            lines = []
            for d in r["detections"]:
                x, y, z, l, w, h, yaw = d["box"]
                # center -> camera frame
                c_cam = T_velo_cam @ np.array([x, y, z, 1.0])
                # KITTI location = BOTTOM center
                loc = c_cam[:3].copy()
                loc[1] += h / 2.0                            # +y is DOWN in cam
                # LiDAR yaw (about +z up) -> camera ry (about +y down):
                # ry = -yaw - pi/2 for the standard KITTI extrinsics; we use
                # the generic relation via the extrinsic z-rotation:
                ry = -(yaw + yaw_T) - np.pi / 2.0
                ry = np.arctan2(np.sin(ry), np.cos(ry))      # wrap to [-pi,pi]
                lines.append(
                    f"{d['label']} 0 0 0 0 0 0 0 "
                    f"{h:.2f} {w:.2f} {l:.2f} "
                    f"{loc[0]:.2f} {loc[1]:.2f} {loc[2]:.2f} {ry:.2f} {d['score']:.3f}")
            idx = r["token"].split("_")[-1]
            with open(os.path.join(label_dir, f"{idx}.txt"), "w") as f:
                f.write("\n".join(lines))

    # ------------------------------------------------------------------
    def _write_nuscenes(self, seq, results):
        """JSON per scene + OpenPCDet-style info entries (keyframes only)."""
        json_out, infos = [], []
        for r in results:
            dets_json = [{
                "sample_data_token": r["token"],
                "translation": [float(v) for v in d["box"][:3]],
                "size": [float(d["box"][4]), float(d["box"][3]), float(d["box"][5])],  # w,l,h (nusc order)
                "yaw": float(d["box"][6]),
                "detection_name": d["label"],
                "detection_score": d["score"],
                "tracking_id": d["track_id"],
            } for d in r["detections"]]
            json_out.append({"token": r["token"], "detections": dets_json})

            if r["is_keyframe"]:
                if r["detections"]:
                    gt_boxes = np.stack([d["box"] for d in r["detections"]]).astype(np.float32)
                else:
                    gt_boxes = np.zeros((0, 7), np.float32)
                infos.append({
                    "lidar_token": r["token"],
                    "gt_boxes": gt_boxes,                    # (N,7) x,y,z,l,w,h,yaw (lidar frame)
                    "gt_names": np.array([d["label"] for d in r["detections"]]),
                    "scores": np.array([d["score"] for d in r["detections"]], np.float32),
                })

        base = os.path.join(self.out_dir, seq["name"])
        with open(base + ".json", "w") as f:
            json.dump(json_out, f)
        with open(base + "_infos.pkl", "wb") as f:
            pickle.dump(infos, f)
