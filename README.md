# ViLGOD — nuScenes & KITTI implementation

Full implementation of **"Vision-Language Guidance for LiDAR-based Unsupervised
3D Object Detection"** (Fruhwirth-Reisinger et al., BMVC 2024), adapted from the
paper's Waymo/Argoverse 2 setup to **nuScenes** and **KITTI**.

```
vilgod/
├── config/                  nuscenes.yaml, kitti.yaml — every hyperparameter
├── data/                    nuscenes_loader.py, kitti_loader.py
├── pipeline/
│   ├── ground_removal.py    Patchwork++ + RANSAC ground plane O_g
│   ├── motion_score.py      PP-score (normalized entropy of neighbor counts)
│   ├── clustering.py        spatio-temporal HDBSCAN [x,y,z,pp,Δt] + filters
│   ├── bbox_fitting.py      L-shape fitting (closeness criterion) + corner utils
│   ├── tracking.py          greedy tracker (1m / relaxed 5m), static/moving rules
│   ├── clip_classifier.py   K=4 view depth maps → CLIP ViT-B/16 zero-shot + voting
│   ├── label_refinement.py  track label propagation, size priors, median boxes
│   └── pseudo_label_writer.py  KITTI .txt / nuScenes JSON + OpenPCDet infos
├── self_training/train_centerpoint.py   CenterPoint config gen + launch
├── evaluate.py              KITTI official eval / nuScenes NDS+mAP + IoU AP
└── run_pipeline.py          end-to-end entry point
```

## 1. Installation

```bash
conda create -n vilgod python=3.10 -y && conda activate vilgod
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### Patchwork++ (ground segmentation)
```bash
sudo apt install libeigen3-dev
git clone https://github.com/url-kaist/patchwork-plusplus
cd patchwork-plusplus && make pyinstall        # installs `pypatchworkpp`
```
If unavailable, the pipeline falls back to a pure-RANSAC ground remover
(works, but worse on slopes/curbs).

### OpenPCDet (self-training + KITTI eval)
```bash
pip install spconv-cu120                       # match your CUDA version
git clone https://github.com/open-mmlab/OpenPCDet /opt/OpenPCDet
cd /opt/OpenPCDet && pip install -r requirements.txt && python setup.py develop
```
Set `self_training.openpcdet_root` in the configs accordingly.

## 2. Data

**nuScenes** — download `v1.0-trainval` (or `v1.0-mini` to test) and point
`dataset.root` at it. The pipeline consumes **all 20 Hz sweeps** for motion
scoring/clustering but exports pseudo-labels only on the 2 Hz keyframes
(where OpenPCDet trains and the official eval runs).

**KITTI** — ViLGOD needs temporal sequences, so discovery runs on the **raw**
drives (`synced+rectified` + oxts). Pseudo-labels are mapped back onto the
object benchmark via the official `train_rand.txt`/`train_mapping.txt`
(set `dataset.mapping_dir`). You need:
- KITTI raw drives → `dataset.raw_root`
- KITTI object benchmark (velodyne, calib, ImageSets) → `dataset.object_root`

## 3. Run

```bash
# 1) generate zero-shot pseudo-labels
python run_pipeline.py --config config/nuscenes.yaml --out outputs/nuscenes
python run_pipeline.py --config config/kitti.yaml    --out outputs/kitti

# 2) self-train CenterPoint (10 epochs, 50% split, no gt-sampling aug)
python self_training/train_centerpoint.py --config config/kitti.yaml --launch
python self_training/train_centerpoint.py --config config/nuscenes.yaml --launch

# 3) evaluate
python evaluate.py --config config/kitti.yaml    --pred-dir <kitti_pred_txt_dir>
python evaluate.py --config config/nuscenes.yaml --pseudo-pkl outputs/nuscenes/pseudo_labels_all.pkl
```

## 4. Key adaptations vs. the paper (Waymo/AV2 → nuScenes/KITTI)

| Topic | Paper | This implementation |
|---|---|---|
| Sequences | WOD/AV2 native sequences | nuScenes scenes (all sweeps); KITTI **raw** drives mapped to the object benchmark |
| Sensor | 64-beam | nuScenes HDL-32E is much sparser → HDBSCAN `min_cluster_size` exposed in config (try 10–12 if far objects are missed) |
| Coordinates | Waymo vehicle frame | unified ego/LiDAR convention (x fwd, y left, z up); KITTI camera-frame conversion happens **only at label export** (velodyne scans are natively in the LiDAR frame) |
| Classes | Vehicle/Ped/Cyclist | full nuScenes 10-class space and KITTI Car/Ped/Cyclist(+Truck), with expanded CLIP prompt sets per class |
| Eval | WOD/AV2 protocols | KITTI official AP (0.7/0.5, E/M/H) via OpenPCDet; nuScenes NDS+mAP via devkit, plus IoU-based BEV AP (0.5 veh / 0.25 ped-cyc) |

Documented assumptions where the paper is ambiguous are marked with
`ASSUMPTION` comments in the code (feature scaling of PP-score/Δt for
HDBSCAN, smooth-motion test, depth-value encoding of the projections).
