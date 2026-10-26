"""
MODULES 7 + 8 — Temporally-coherent label & bounding box refinement.

LABEL refinement (paper Sec. 3.2 / Appendix A):
    For each track, propagate the most confident CLIP label to ALL of the
    track's boxes iff
        * that label appears in >= 60% of the track's classified frames, AND
        * its best score exceeds 0.5 (vehicle classes) / 0.3 (all others).
    For MOVING objects failing this test, fall back to size-prior
    classification (dataset-specific tables in the config); static objects
    failing the test keep their raw per-frame CLIP prediction.
    Tracks resolved to 'background' are dropped.

BOX refinement (paper Sec. 3.2 / Appendix A):
    * median box: take the M box candidates with the most cluster points,
      use the per-dimension median (l, w, h);
    * STATIC tracks: propagate the median box at the MEDIAN position with
      majority-vote orientation;
    * MOVING tracks: propagate the median box size along the track,
      keeping per-frame yaw, then re-anchor each box at the corner closest
      to the ego vehicle (OYSTER corner alignment) — this corrects the
      center bias caused by partial (2.5D) observations;
    * inflate every final box by 0.3 m per dimension (annotation bias);
    * final sanity size filter: 0.2<w<3.5, 0.2<l<20.0, 0.5<h<4.0.
"""

import numpy as np

from .bbox_fitting import closest_corner, align_box_to_corner


def _yaw_majority(yaws, bins=36):
    """Majority-vote orientation: histogram over [0, pi) (yaw is pi-symmetric
    for L-shape boxes), return circular mean of the winning bin."""
    y = np.mod(np.asarray(yaws), np.pi)
    hist, edges = np.histogram(y, bins=bins, range=(0, np.pi))
    b = hist.argmax()
    sel = y[(y >= edges[b]) & (y < edges[b + 1])]
    return float(sel.mean()) if len(sel) else float(y.mean())


class TemporalRefiner:
    """Track-level label propagation and median-box refinement."""

    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.dataset = dataset.lower()
        self.vehicle_classes = set(cfg.get("vehicle_classes", []))

    # ------------------------------------------------------------------
    def _reliable_track_label(self, track):
        """Return (label, score) if the track has a reliable CLIP label."""
        labeled = [d for d in track.detections if d.clip_label is not None]
        if not labeled:
            return None
        labels = [d.clip_label for d in labeled]
        uniq, counts = np.unique(labels, return_counts=True)
        order = np.argsort(-counts)
        for i in order:
            label = uniq[i]
            ratio = counts[i] / len(labeled)
            if ratio < self.cfg.get("label_min_track_ratio", 0.60):
                break  # sorted by count: no later label can pass either
            best_score = max(d.clip_score for d in labeled if d.clip_label == label)
            thr = (self.cfg.get("score_threshold_vehicle", 0.5)
                   if label in self.vehicle_classes
                   else self.cfg.get("score_threshold_other", 0.3))
            if best_score > thr:
                return label, best_score
        return None

    def _size_prior_label(self, box):
        """Size-based fallback classification (moving objects only)."""
        l, w, h = box[3], box[4], box[5]
        # length/width of an L-shape box are interchangeable for thin
        # objects -> compare against (max, min) consistently
        L, W = max(l, w), min(l, w)
        for cls, pr in self.cfg["size_priors"].items():
            if (pr["w"][0] < W < pr["w"][1]
                    and pr["l"][0] < L < pr["l"][1]
                    and pr["h"][0] < h < pr["h"][1]):
                return cls
        return "background"

    # ------------------------------------------------------------------
    def refine(self, tracks, num_frames, ego_positions):
        """
        Args:
            tracks:        list[tracking.Track]
            num_frames:    sequence length
            ego_positions: (T,3) ego/sensor origin per frame (GLOBAL frame)
        Returns:
            dict frame_idx -> list of dicts
                {'box': (7,), 'label': str, 'score': float, 'track_id': int}
            Boxes are in the GLOBAL frame; the writer transforms them back.
        """
        out = {t: [] for t in range(num_frames)}
        f = self.cfg["final_size_filter"]
        inflate = self.cfg.get("inflate", 0.3)
        M = self.cfg.get("median_top_m", 5)

        for tr in tracks:
            # -------------------------------------------- 7. label
            rel = self._reliable_track_label(tr)
            if rel is not None:
                track_label, track_score = rel
                per_det_labels = None                     # propagate one label
            elif not tr.is_static:
                # moving object without reliable CLIP label -> size priors
                ref_box = max(tr.detections, key=lambda d: d.num_points).box
                track_label = self._size_prior_label(ref_box)
                track_score = 0.10                        # low-confidence marker
                per_det_labels = None
            else:
                # static & unreliable: keep raw per-frame CLIP predictions
                track_label, track_score = None, None
                per_det_labels = [(d.clip_label, d.clip_score) for d in tr.detections]

            # -------------------------------------------- 8. boxes
            top = sorted(tr.detections, key=lambda d: -d.num_points)[:M]
            dims = np.stack([d.box[3:6] for d in top])
            med_dims = np.median(dims, axis=0)

            if tr.is_static:
                centers = np.stack([d.box[:3] for d in tr.detections])
                med_center = np.median(centers, axis=0)
                yaw = _yaw_majority([d.box[6] for d in tr.detections])
                proto = np.array([*med_center, *med_dims, yaw])
                boxes = {d.frame_idx: proto.copy() for d in tr.detections}
            else:
                boxes = {}
                for d in tr.detections:
                    b = d.box.copy()
                    # OYSTER corner alignment: remember the observed box's
                    # corner closest to the ego vehicle, resize to the
                    # median dims, then translate so that corner is fixed.
                    ego_xy = ego_positions[d.frame_idx][:2]
                    anchor_xy, ci = closest_corner(b, ego_xy)
                    b[3:6] = med_dims
                    b = align_box_to_corner(b, anchor_xy, ci)
                    boxes[d.frame_idx] = b

            # -------------------------------------------- emit
            for i, d in enumerate(tr.detections):
                b = boxes[d.frame_idx]
                if per_det_labels is None:
                    label, score = track_label, track_score
                else:
                    label, score = per_det_labels[i]
                if label is None or label == "background":
                    continue
                bb = b.copy()
                bb[3:6] += inflate                       # inflate l,w,h by 0.3m
                L, W, H = bb[3], bb[4], bb[5]
                if not (f["w"][0] < W < f["w"][1]
                        and f["l"][0] < L < f["l"][1]
                        and f["h"][0] < H < f["h"][1]):
                    continue                              # clearly wrong box
                out[d.frame_idx].append({
                    "box": bb, "label": label,
                    "score": float(score), "track_id": tr.id,
                })
        return out
