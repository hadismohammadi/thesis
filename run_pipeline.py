"""
ViLGOD — Vision-Language Guidance for LiDAR-based Unsupervised 3D Object Detection
(Fruhwirth-Reisinger et al., BMVC 2024), adapted for nuScenes and KITTI.

End-to-end entry point. Per LiDAR sequence the pipeline runs:

  1.  Ground removal               (Patchwork++ + RANSAC ground plane)   [pipeline/ground_removal.py]
  2.  Motion scoring               (PP-score over adjacent frames)       [pipeline/motion_score.py]
  3.  Spatio-temporal clustering   (HDBSCAN on [x,y,z,pp,dt] features)   [pipeline/clustering.py]
  4.  Bounding box fitting         (L-shape + corner alignment)          [pipeline/bbox_fitting.py]
  5.  Multi-target tracking        (greedy assignment, static/moving)    [pipeline/tracking.py]
  6.  CLIP zero-shot classification(multi-view depth maps)               [pipeline/clip_classifier.py]
  7.  Temporal label refinement    (track-level label propagation)       [pipeline/label_refinement.py]
  8.  Temporal bbox refinement     (median box propagation, inflation)   [pipeline/label_refinement.py]
  9.  Pseudo-label export          (OpenPCDet / KITTI / nuScenes format) [pipeline/pseudo_label_writer.py]
  10. Self-training                (CenterPoint via OpenPCDet)           [self_training/train_centerpoint.py]

Usage:
    python run_pipeline.py --config config/nuscenes.yaml --out outputs/nuscenes
    python run_pipeline.py --config config/kitti.yaml    --out outputs/kitti
"""

import argparse
import os
import pickle
import time

import numpy as np
import yaml

from data.nuscenes_loader import NuScenesSequenceLoader
from data.kitti_loader import KittiSequenceLoader
from pipeline.ground_removal import GroundRemover
from pipeline.motion_score import PPScoreComputer
from pipeline.clustering import SpatioTemporalClusterer
from pipeline.bbox_fitting import BoxFitter
from pipeline.tracking import GreedyTracker
from pipeline.clip_classifier import ClipClassifier
from pipeline.label_refinement import TemporalRefiner
from pipeline.pseudo_label_writer import PseudoLabelWriter


def load_config(path):
    """Load a YAML configuration file into a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_loader(cfg):
    """Instantiate the correct dataset loader from the config."""
    name = cfg["dataset"]["name"].lower()
    if name == "nuscenes":
        return NuScenesSequenceLoader(cfg)
    if name == "kitti":
        return KittiSequenceLoader(cfg)
    raise ValueError(f"Unknown dataset: {name}")


def process_sequence(seq, cfg, ground, pp, clusterer, fitter, clip_clf, refiner, writer):
    """
    Run the full unsupervised detection pipeline on a single LiDAR sequence.

    Args:
        seq:     dict with keys
                 'frames':  list of dicts {'points': (N,3/4) float32 in SENSOR frame,
                                           'pose':   (4,4) sensor->global,
                                           'token':  frame identifier,
                                           'timestamp': float seconds}
                 'name':    sequence name
        cfg:     full config dict
        ...      pipeline component instances

    Returns:
        per-frame pseudo-label dicts (also written to disk by `writer`).
    """
    frames = seq["frames"]
    T = len(frames)
    print(f"[{seq['name']}] {T} frames")

    # ---------------------------------------------------------------- 1. ground removal
    # Non-ground points per frame, plus a RANSAC ground plane O_g per frame.
    nonground_sensor = []      # points in sensor frame (for label export / projection)
    nonground_global = []      # points in global frame (for PP-score & clustering)
    ground_planes = []         # (4,) plane [a,b,c,d] in SENSOR frame: ax+by+cz+d=0
    for fr in frames:
        ng_pts, plane = ground.remove(fr["points"][:, :3])
        nonground_sensor.append(ng_pts)
        hom = np.hstack([ng_pts, np.ones((len(ng_pts), 1))])
        nonground_global.append((fr["pose"] @ hom.T).T[:, :3].astype(np.float32))
        ground_planes.append(plane)

    # ---------------------------------------------------------------- 2. PP-score
    # Per-point persistence score in [0,1]; LOW score == ephemeral == moving.
    pp_scores = pp.compute_sequence(nonground_global)

    # ---------------------------------------------------------------- 3-4. clustering + boxes
    # For every reference frame, aggregate a temporal window, cluster, filter,
    # fit boxes and classify each box static/moving from the PP-scores.
    detections = {}            # frame_idx -> list of Detection
    n_agg = cfg["clustering"]["num_agg_frames"]
    for t in range(T):
        window = list(range(t, min(t + n_agg, T)))
        segments = clusterer.cluster(
            frames_global=[nonground_global[j] for j in window],
            frames_pp=[pp_scores[j] for j in window],
            timestamps=[frames[j]["timestamp"] for j in window],
            ref_idx=0,
            ground_plane_sensor=ground_planes[t],
            ref_pose=frames[t]["pose"],
        )
        dets = []
        for seg in segments:
            det = fitter.fit(seg)                      # box in GLOBAL frame
            if det is None:
                continue
            det.frame_idx = t
            dets.append(det)
        detections[t] = dets

    # ---------------------------------------------------------------- 5. tracking
    tracker = GreedyTracker(cfg["tracking"])
    tracks = tracker.run(detections, num_frames=T)
    print(f"[{seq['name']}] {len(tracks)} tracks")

    # ---------------------------------------------------------------- 6. CLIP classification
    # Classify every detection (cluster segment) with multi-view depth maps.
    clip_clf.classify_tracks(tracks)

    # ---------------------------------------------------------------- 7-8. temporal refinement
    # Track-level label propagation, size-prior fallback for moving objects,
    # median-box propagation, corner alignment, inflation, final size filter.
    ego_positions = np.stack([fr["pose"][:3, 3] for fr in frames])
    final_dets = refiner.refine(tracks, num_frames=T, ego_positions=ego_positions)

    # ---------------------------------------------------------------- 9. export
    results = writer.write_sequence(seq, final_dets)
    return results


def main():
    parser = argparse.ArgumentParser(description="ViLGOD pseudo-label generation")
    parser.add_argument("--config", required=True, help="config/nuscenes.yaml or config/kitti.yaml")
    parser.add_argument("--out", required=True, help="output directory for pseudo-labels")
    parser.add_argument("--max-sequences", type=int, default=-1, help="limit #sequences (debug)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["output_dir"] = args.out
    os.makedirs(args.out, exist_ok=True)

    loader = build_loader(cfg)
    ground = GroundRemover(cfg["ground_removal"])
    pp = PPScoreComputer(cfg["motion_score"])
    clusterer = SpatioTemporalClusterer(cfg["clustering"])
    fitter = BoxFitter(cfg["bbox_fitting"])
    clip_clf = ClipClassifier(cfg["clip"])
    refiner = TemporalRefiner(cfg["refinement"], dataset=cfg["dataset"]["name"])
    writer = PseudoLabelWriter(cfg)

    t0 = time.time()
    all_results = {}
    for i, seq in enumerate(loader):
        if 0 <= args.max_sequences <= i:
            break
        all_results[seq["name"]] = process_sequence(
            seq, cfg, ground, pp, clusterer, fitter, clip_clf, refiner, writer
        )

    with open(os.path.join(args.out, "pseudo_labels_all.pkl"), "wb") as f:
        pickle.dump(all_results, f)
    print(f"Done in {time.time() - t0:.1f}s. Pseudo-labels written to {args.out}")
    print("Next step: python self_training/train_centerpoint.py --config", args.config)


if __name__ == "__main__":
    main()
