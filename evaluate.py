"""
Evaluation of pseudo-labels (zero-shot ViLGOD) or trained CenterPoint
predictions on the validation split.

KITTI:
    Uses OpenPCDet's official `kitti_object_eval_python` — AP at IoU 0.7
    (Car) / 0.5 (Pedestrian, Cyclist), Easy/Moderate/Hard (R40).
    Predictions must be KITTI-format .txt files in --pred-dir.

nuScenes:
    (a) official protocol: if predictions are exported in the nuScenes
        result-JSON format, runs nuscenes.eval.detection (NDS + mAP);
    (b) additionally, an IoU-based BEV/3D AP evaluation at the thresholds
        from the config (0.5 vehicles / 0.25 ped+cyclists), implemented
        here, for comparability with the paper's evaluation style.

Usage:
    python evaluate.py --config config/kitti.yaml    --pred-dir outputs/kitti/pseudo_labels_object
    python evaluate.py --config config/nuscenes.yaml --pred-json outputs/nuscenes/results_nusc.json
"""

import argparse
import glob
import os
import sys

import numpy as np
import yaml


# ----------------------------------------------------------------------
# Generic IoU-based AP (BEV) used for the nuScenes adaptation
# ----------------------------------------------------------------------
def rotated_bev_iou(box_a, boxes_b):
    """Exact rotated BEV IoU via shapely polygons. box: (x,y,z,l,w,h,yaw)."""
    from shapely.geometry import Polygon

    def poly(b):
        cx, cy, _, l, w, _, yaw = b
        c, s = np.cos(yaw), np.sin(yaw)
        pts = [(dx * c - dy * s + cx, dx * s + dy * c + cy)
               for dx, dy in [(l/2, w/2), (l/2, -w/2), (-l/2, -w/2), (-l/2, w/2)]]
        return Polygon(pts)

    pa = poly(box_a)
    out = np.zeros(len(boxes_b))
    for i, b in enumerate(boxes_b):
        pb = poly(b)
        inter = pa.intersection(pb).area
        union = pa.area + pb.area - inter
        out[i] = inter / union if union > 0 else 0.0
    return out


def average_precision(gt_by_frame, pred_by_frame, iou_thr):
    """
    Standard 11-point-free AP: rank all predictions by score, greedy match
    against unmatched GT with rotated BEV IoU >= iou_thr.

    Args:
        gt_by_frame:   dict frame -> (G,7) array
        pred_by_frame: dict frame -> list of (box(7,), score)
    """
    preds = []
    for fr, dets in pred_by_frame.items():
        for box, score in dets:
            preds.append((score, fr, box))
    preds.sort(key=lambda x: -x[0])
    n_gt = sum(len(g) for g in gt_by_frame.values())
    if n_gt == 0:
        return 0.0

    matched = {fr: np.zeros(len(g), bool) for fr, g in gt_by_frame.items()}
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, (score, fr, box) in enumerate(preds):
        gts = gt_by_frame.get(fr, np.zeros((0, 7)))
        if len(gts) == 0:
            fp[i] = 1
            continue
        ious = rotated_bev_iou(box, gts)
        ious[matched[fr]] = -1.0
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            tp[i] = 1
            matched[fr][j] = True
        else:
            fp[i] = 1

    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c / n_gt
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    # precision envelope + integration
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    idx = np.where(np.diff(np.concatenate([[0], recall])) > 0)[0]
    return float(np.sum(precision[idx] * np.diff(np.concatenate([[0], recall]))[idx]))


# ----------------------------------------------------------------------
# KITTI: official eval via OpenPCDet
# ----------------------------------------------------------------------
def parse_kitti_label(path, with_score):
    boxes, names, scores = [], [], []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) < 15:
                continue
            names.append(p[0])
            h, w, l = float(p[8]), float(p[9]), float(p[10])
            x, y, z, ry = float(p[11]), float(p[12]), float(p[13]), float(p[14])
            boxes.append([x, y, z, l, h, w, ry])
            scores.append(float(p[15]) if with_score and len(p) > 15 else 1.0)
    return names, np.array(boxes).reshape(-1, 7), np.array(scores)


def evaluate_kitti(cfg, pred_dir):
    pcdet = cfg["self_training"]["openpcdet_root"]
    sys.path.insert(0, pcdet)
    from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
    from pcdet.datasets.kitti.kitti_object_eval_python import kitti_common as kitti

    gt_dir = os.path.join(cfg["dataset"]["object_root"], "training", "label_2")
    val_split = os.path.join(cfg["dataset"]["object_root"], "ImageSets", "val.txt")
    ids = [int(x) for x in open(val_split).read().split()]

    gt_annos = kitti.get_label_annos(gt_dir, ids)
    dt_annos = kitti.get_label_annos(pred_dir, ids)
    classes = ["Car", "Pedestrian", "Cyclist"]
    result, _ = kitti_eval.get_official_eval_result(gt_annos, dt_annos, classes)
    print(result)


# ----------------------------------------------------------------------
# nuScenes: official NDS/mAP + adapted IoU-based AP
# ----------------------------------------------------------------------
def evaluate_nuscenes(cfg, pred_json):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.detection.config import config_factory

    d = cfg["dataset"]
    nusc = NuScenes(version=d["version"], dataroot=d["root"], verbose=False)
    eval_cfg = config_factory("detection_cvpr_2019")
    nusc_eval = DetectionEval(nusc, config=eval_cfg, result_path=pred_json,
                              eval_set="val", output_dir=os.path.dirname(pred_json),
                              verbose=True)
    metrics, _ = nusc_eval.evaluate()
    print(f"NDS: {metrics.nd_score:.4f}  mAP: {metrics.mean_ap:.4f}")


def evaluate_pseudo_labels_iou(cfg, pseudo_pkl):
    """IoU-based BEV AP of zero-shot pseudo-labels vs. GT (nuScenes)."""
    import pickle
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion

    d = cfg["dataset"]
    nusc = NuScenes(version=d["version"], dataroot=d["root"], verbose=False)
    with open(pseudo_pkl, "rb") as f:
        all_results = pickle.load(f)

    gt_by_frame = {c: {} for c in cfg["evaluation"]["iou_thresholds"]}
    pred_by_frame = {c: {} for c in cfg["evaluation"]["iou_thresholds"]}

    for seq_name, frames in all_results.items():
        for fr in frames:
            if not fr["is_keyframe"]:
                continue
            sd = nusc.get("sample_data", fr["token"])
            sample = nusc.get("sample", sd["sample_token"])
            # GT boxes in the lidar/ego frame
            for cls in gt_by_frame:
                gt_by_frame[cls].setdefault(fr["token"], [])
                pred_by_frame[cls].setdefault(fr["token"], [])
            _, gt_boxes, _ = nusc.get_sample_data(sd["token"])
            for b in gt_boxes:
                name = nusc.get("sample_annotation", b.token)["category_name"]
                cls = _nusc_category_to_detection(name)
                if cls in gt_by_frame:
                    yaw = Quaternion(matrix=b.rotation_matrix).yaw_pitch_roll[0]
                    gt_by_frame[cls][fr["token"]].append(
                        [b.center[0], b.center[1], b.center[2],
                         b.wlh[1], b.wlh[0], b.wlh[2], yaw])
            for det in fr["detections"]:
                cls = det["label"]
                if cls in pred_by_frame:
                    pred_by_frame[cls][fr["token"]].append((det["box"], det["score"]))

    for cls, thr in cfg["evaluation"]["iou_thresholds"].items():
        gtf = {k: np.array(v).reshape(-1, 7) for k, v in gt_by_frame[cls].items()}
        ap = average_precision(gtf, pred_by_frame[cls], thr)
        print(f"{cls:>22s}  AP_BEV@{thr}: {ap:.4f}")


def _nusc_category_to_detection(name):
    m = {"vehicle.car": "car", "vehicle.truck": "truck", "vehicle.bus": "bus",
         "vehicle.trailer": "trailer",
         "vehicle.construction": "construction_vehicle",
         "human.pedestrian": "pedestrian", "vehicle.motorcycle": "motorcycle",
         "vehicle.bicycle": "bicycle", "movable_object.trafficcone": "traffic_cone",
         "movable_object.barrier": "barrier"}
    for k, v in m.items():
        if name.startswith(k):
            return v
    return None


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pred-dir", help="KITTI: dir of predicted .txt files")
    parser.add_argument("--pred-json", help="nuScenes: results json (official eval)")
    parser.add_argument("--pseudo-pkl", help="nuScenes: pseudo_labels_all.pkl (IoU AP)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if cfg["dataset"]["name"].lower() == "kitti":
        assert args.pred_dir, "--pred-dir required for KITTI"
        evaluate_kitti(cfg, args.pred_dir)
    else:
        if args.pred_json:
            evaluate_nuscenes(cfg, args.pred_json)
        if args.pseudo_pkl:
            evaluate_pseudo_labels_iou(cfg, args.pseudo_pkl)


if __name__ == "__main__":
    main()
