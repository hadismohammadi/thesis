"""
MODULE 6 — CLIP zero-shot classification of point cluster segments.

Pipeline per segment (ViLGOD Sec. 3.2, following PointCLIP V2 [74]):
    1. zero-center the segment's 3D points (reference SENSOR frame, so the
       viewing direction matches what the LiDAR actually observed);
    2. generate K=4 views by small rotations:
           view 1: identity, view 2: yaw +18deg, view 3: yaw -18deg,
           view 4: pitch +6deg (looking slightly down from above; tilting
           the other way is useless — the ground plane means objects have
           no valid bottom shape);
    3. orthographic projection onto a coarse grid (voxelization), keeping
       the NEAREST depth per pixel; densification by morphological closing;
       Gaussian smoothing; upsample to 224x224 -> "natural-looking" depth map;
    4. CLIP ViT-B/16 visual embedding vs. precomputed text embeddings of the
       EXPANDED category prompts ("a point representation of a <class>" —
       best template in the paper's ablation, Tab. 8);
    5. fine predictions are merged back onto the coarse dataset classes;
    6. multi-view voting: majority class over K views, score = mean
       similarity of agreeing views; ties broken by higher mean score.

Install: pip install git+https://github.com/openai/CLIP.git
"""

import numpy as np
import cv2
import torch

try:
    import clip as openai_clip
    _HAS_CLIP = True
except ImportError:
    _HAS_CLIP = False

# CLIP's normalization constants (RGB)
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def _rotation(yaw_deg, pitch_deg):
    """Rotation matrix: yaw about z (up), pitch about y (left)."""
    y, p = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    return Ry @ Rz


def project_depth_map(points, yaw_deg, pitch_deg, grid=112, out_size=224,
                      densify_kernel=3, gaussian_sigma=2.0):
    """
    Render a smooth 2D depth map of zero-centered points.

    Sensor-frame convention: x=forward (viewing/depth axis), y=left, z=up.
    Image axes: u = -y (so the object's left appears on the image's left
    when viewed from the sensor), v = -z (image rows grow downward).
    Pixel intensity encodes CLOSENESS: nearer surface -> brighter, which
    matches the depth-map renderings CLIP saw via PointCLIP V2.

    Returns:
        (out_size, out_size) float32 image in [0, 1].
    """
    pts = points - points.mean(0, keepdims=True)        # zero-center
    pts = pts @ _rotation(yaw_deg, pitch_deg).T

    u, v, d = -pts[:, 1], -pts[:, 2], pts[:, 0]

    # normalize the object into the image with 10% margin, preserving aspect
    span = max(u.max() - u.min(), v.max() - v.min(), 1e-3)
    scale = (grid * 0.8) / span
    ui = ((u - 0.5 * (u.max() + u.min())) * scale + grid / 2.0).astype(int)
    vi = ((v - 0.5 * (v.max() + v.min())) * scale + grid / 2.0).astype(int)
    ok = (ui >= 0) & (ui < grid) & (vi >= 0) & (vi < grid)
    ui, vi, d = ui[ok], vi[ok], d[ok]
    if len(d) == 0:
        return np.zeros((out_size, out_size), np.float32)

    # closeness in [0.3, 1] (0 reserved for background)
    d_span = max(d.max() - d.min(), 1e-3)
    val = 0.3 + 0.7 * (d.max() - d) / d_span

    img = np.zeros((grid, grid), np.float32)
    # keep nearest (max closeness) per pixel
    np.maximum.at(img, (vi, ui), val.astype(np.float32))

    # densification: grayscale morphological closing fills small holes
    k = np.ones((densify_kernel, densify_kernel), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    img = cv2.dilate(img, k)

    # smoothing + upsampling
    img = cv2.GaussianBlur(img, (0, 0), gaussian_sigma)
    img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    return np.clip(img, 0.0, 1.0)


class ClipClassifier:
    """Multi-view CLIP zero-shot classifier for LiDAR cluster segments."""

    def __init__(self, cfg):
        if not _HAS_CLIP:
            raise ImportError(
                "OpenAI CLIP not installed: pip install git+https://github.com/openai/CLIP.git")
        self.cfg = cfg
        self.device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        self.model, _ = openai_clip.load(cfg.get("model_name", "ViT-B/16"),
                                         device=self.device)
        self.model.eval()
        self.K = cfg.get("num_views", 4)
        self.yaws = cfg.get("view_yaws_deg", [0, 18, -18, 0])
        self.pitches = cfg.get("view_pitches_deg", [0, 0, 0, 6])
        self.batch_size = cfg.get("batch_size", 64)
        self.max_per_track = cfg.get("max_segments_per_track", 10)

        # ---------------- precompute text embeddings (expanded categories)
        template = cfg.get("prompt_template", "a point representation of a {}")
        self.fine_to_coarse = []       # fine index -> coarse class name
        prompts = []
        for coarse, fine_names in cfg["categories"].items():
            for name in fine_names:
                prompts.append(template.format(name))
                self.fine_to_coarse.append(coarse)
        with torch.no_grad():
            tokens = openai_clip.tokenize(prompts).to(self.device)
            txt = self.model.encode_text(tokens).float()
            self.text_emb = txt / txt.norm(dim=-1, keepdim=True)   # (F, D)
        self.coarse_classes = list(cfg["categories"].keys())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_images(self, imgs):
        """Encode a (B, 224, 224) float batch of depth maps -> (B, D)."""
        x = np.repeat(imgs[:, None], 3, axis=1)                       # grayscale -> RGB
        x = (x - _CLIP_MEAN[None, :, None, None]) / _CLIP_STD[None, :, None, None]
        x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        z = self.model.encode_image(x).float()
        return z / z.norm(dim=-1, keepdim=True)

    def classify_segment_batch(self, segments):
        """
        Classify a list of Segments.

        Returns:
            list of (coarse_label, score) — multi-view voting result.
        """
        c = self.cfg
        all_imgs = []
        for seg in segments:
            for k in range(self.K):
                all_imgs.append(project_depth_map(
                    seg.points_sensor, self.yaws[k], self.pitches[k],
                    grid=c.get("proj_grid", 112), out_size=c.get("image_size", 224),
                    densify_kernel=c.get("densify_kernel", 3),
                    gaussian_sigma=c.get("gaussian_sigma", 2.0)))
        all_imgs = np.stack(all_imgs)                                  # (S*K, H, W)

        embs = []
        for i in range(0, len(all_imgs), self.batch_size):
            embs.append(self._encode_images(all_imgs[i:i + self.batch_size]))
        embs = torch.cat(embs)                                         # (S*K, D)

        sims = (embs @ self.text_emb.T).cpu().numpy()                  # (S*K, F)

        results = []
        F2C = np.array([self.coarse_classes.index(c_) for c_ in self.fine_to_coarse])
        for s in range(len(segments)):
            view_sims = sims[s * self.K:(s + 1) * self.K]              # (K, F)
            # merge fine -> coarse by taking max fine similarity per coarse class
            coarse_sims = np.full((self.K, len(self.coarse_classes)), -np.inf, np.float32)
            for ci in range(len(self.coarse_classes)):
                mask = F2C == ci
                coarse_sims[:, ci] = view_sims[:, mask].max(1)
            view_pred = coarse_sims.argmax(1)                          # (K,)

            # ---- multi-view voting
            votes = np.bincount(view_pred, minlength=len(self.coarse_classes))
            best = votes.max()
            tied = np.flatnonzero(votes == best)
            if len(tied) == 1:
                winner = tied[0]
            else:  # tie-break: class with the highest mean score among voters
                means = []
                for ci in tied:
                    agree = view_pred == ci
                    means.append(coarse_sims[agree, ci].mean())
                winner = tied[int(np.argmax(means))]
            agree = view_pred == winner
            score = float(coarse_sims[agree, winner].mean())
            results.append((self.coarse_classes[winner], score))
        return results

    # ------------------------------------------------------------------
    def classify_tracks(self, tracks):
        """
        Classify (a subsample of) each track's detections in-place,
        setting det.clip_label / det.clip_score. Unclassified detections
        keep clip_label=None and are filled by the temporal refinement.
        """
        to_classify = []
        for tr in tracks:
            dets = tr.detections
            if len(dets) > self.max_per_track:
                idx = np.linspace(0, len(dets) - 1, self.max_per_track).astype(int)
                dets = [dets[i] for i in idx]
            to_classify.extend(dets)
        if not to_classify:
            return
        # chunked to bound peak memory on long sequences
        chunk = 256
        for i in range(0, len(to_classify), chunk):
            block = to_classify[i:i + chunk]
            results = self.classify_segment_batch([d.segment for d in block])
            for det, (label, score) in zip(block, results):
                det.clip_label = label
                det.clip_score = score
