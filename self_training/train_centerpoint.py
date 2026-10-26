"""
MODULE 9 (part 2) — Self-training with CenterPoint (OpenPCDet).

Generates an OpenPCDet model config matching the paper's protocol and
launches training:
    * Adam + one-cycle LR policy, max_lr = 0.003
    * 10 epochs
    * 50% of the training split (random subsample of infos, fixed seed)
    * NO pseudo-label ("gt sampling") database augmentation
    * single round of self-training (no repetition)

Install OpenPCDet:
    git clone https://github.com/open-mmlab/OpenPCDet /opt/OpenPCDet
    cd /opt/OpenPCDet && pip install -r requirements.txt
    python setup.py develop          # builds CUDA ops (spconv required:
                                     # pip install spconv-cu120, match your CUDA)

KITTI:  this script merges the per-drive pseudo-label .txt files into the
        object-benchmark layout via the official raw<->object mapping
        (train_rand.txt / train_mapping.txt), then rebuilds OpenPCDet's
        kitti infos with `pcdet.datasets.kitti.kitti_dataset`.
nuScenes: replaces gt_boxes/gt_names inside OpenPCDet's
        nuscenes_infos_10sweeps_train.pkl with our pseudo-labels (matched
        by lidar sample_data token).

Usage:
    python self_training/train_centerpoint.py --config config/kitti.yaml
"""

import argparse
import os
import pickle
import random
import shutil
import subprocess
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ----------------------------------------------------------------------
# OpenPCDet config templates (CenterPoint), paper training protocol
# ----------------------------------------------------------------------
CENTERPOINT_KITTI_CFG = """\
CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']

DATA_CONFIG:
    _BASE_CONFIG_: {pcdet}/tools/cfgs/dataset_configs/kitti_dataset.yaml
    DATA_AUGMENTOR:
        DISABLE_AUG_LIST: ['placeholder']
        AUG_CONFIG_LIST:
            # NOTE: NO gt_sampling (no pseudo-label database augmentation)
            - NAME: random_world_flip
              ALONG_AXIS_LIST: ['x']
            - NAME: random_world_rotation
              WORLD_ROT_ANGLE: [-0.78539816, 0.78539816]
            - NAME: random_world_scaling
              WORLD_SCALE_RANGE: [0.95, 1.05]

MODEL:
    NAME: CenterPoint
    VFE: {{NAME: MeanVFE}}
    BACKBONE_3D: {{NAME: VoxelResBackBone8x}}
    MAP_TO_BEV: {{NAME: HeightCompression, NUM_BEV_FEATURES: 256}}
    BACKBONE_2D:
        NAME: BaseBEVBackbone
        LAYER_NUMS: [5, 5]
        LAYER_STRIDES: [1, 2]
        NUM_FILTERS: [128, 256]
        UPSAMPLE_STRIDES: [1, 2]
        NUM_UPSAMPLE_FILTERS: [256, 256]
    DENSE_HEAD:
        NAME: CenterHead
        CLASS_AGNOSTIC: False
        CLASS_NAMES_EACH_HEAD: [['Car', 'Pedestrian', 'Cyclist']]
        SHARED_CONV_CHANNEL: 64
        USE_BIAS_BEFORE_NORM: True
        NUM_HM_CONV: 2
        SEPARATE_HEAD_CFG:
            HEAD_ORDER: ['center', 'center_z', 'dim', 'rot']
            HEAD_DICT: {{
                'center': {{'out_channels': 2, 'num_conv': 2}},
                'center_z': {{'out_channels': 1, 'num_conv': 2}},
                'dim': {{'out_channels': 3, 'num_conv': 2}},
                'rot': {{'out_channels': 2, 'num_conv': 2}},
            }}
        TARGET_ASSIGNER_CONFIG:
            FEATURE_MAP_STRIDE: 8
            NUM_MAX_OBJS: 500
            GAUSSIAN_OVERLAP: 0.1
            MIN_RADIUS: 2
        LOSS_CONFIG:
            LOSS_WEIGHTS: {{'cls_weight': 1.0, 'loc_weight': 2.0,
                            'code_weights': [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]}}
        POST_PROCESSING:
            SCORE_THRESH: 0.1
            POST_CENTER_LIMIT_RANGE: [-75.2, -75.2, -2, 75.2, 75.2, 4]
            MAX_OBJ_PER_SAMPLE: 500
            NMS_CONFIG:
                NMS_TYPE: nms_gpu
                NMS_THRESH: 0.7
                NMS_PRE_MAXSIZE: 4096
                NMS_POST_MAXSIZE: 500
    POST_PROCESSING:
        RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
        EVAL_METRIC: kitti

OPTIMIZATION:
    BATCH_SIZE_PER_GPU: {batch_size}
    NUM_EPOCHS: {epochs}
    OPTIMIZER: adam_onecycle
    LR: {max_lr}
    WEIGHT_DECAY: 0.01
    MOMENTUM: 0.9
    MOMS: [0.95, 0.85]
    PCT_START: 0.4
    DIV_FACTOR: 10
    DECAY_STEP_LIST: [35, 45]
    LR_DECAY: 0.1
    LR_CLIP: 0.0000001
    LR_WARMUP: False
    WARMUP_EPOCH: 1
    GRAD_NORM_CLIP: 10
"""

CENTERPOINT_NUSC_CFG = """\
CLASS_NAMES: ['car','truck','construction_vehicle','bus','trailer',
              'barrier','motorcycle','bicycle','pedestrian','traffic_cone']

DATA_CONFIG:
    _BASE_CONFIG_: {pcdet}/tools/cfgs/dataset_configs/nuscenes_dataset.yaml
    INFO_PATH: {{'train': [{train_info}], 'test': ['nuscenes_infos_10sweeps_val.pkl']}}
    DATA_AUGMENTOR:
        DISABLE_AUG_LIST: ['placeholder']
        AUG_CONFIG_LIST:
            # NOTE: NO gt_sampling (no pseudo-label database augmentation)
            - NAME: random_world_flip
              ALONG_AXIS_LIST: ['x', 'y']
            - NAME: random_world_rotation
              WORLD_ROT_ANGLE: [-0.78539816, 0.78539816]
            - NAME: random_world_scaling
              WORLD_SCALE_RANGE: [0.9, 1.1]

MODEL:
    NAME: CenterPoint
    VFE: {{NAME: MeanVFE}}
    BACKBONE_3D: {{NAME: VoxelResBackBone8x}}
    MAP_TO_BEV: {{NAME: HeightCompression, NUM_BEV_FEATURES: 256}}
    BACKBONE_2D:
        NAME: BaseBEVBackbone
        LAYER_NUMS: [5, 5]
        LAYER_STRIDES: [1, 2]
        NUM_FILTERS: [128, 256]
        UPSAMPLE_STRIDES: [1, 2]
        NUM_UPSAMPLE_FILTERS: [256, 256]
    DENSE_HEAD:
        NAME: CenterHead
        CLASS_AGNOSTIC: False
        CLASS_NAMES_EACH_HEAD: [
            ['car'], ['truck','construction_vehicle'], ['bus','trailer'],
            ['barrier'], ['motorcycle','bicycle'], ['pedestrian','traffic_cone']]
        SHARED_CONV_CHANNEL: 64
        USE_BIAS_BEFORE_NORM: True
        NUM_HM_CONV: 2
        SEPARATE_HEAD_CFG:
            HEAD_ORDER: ['center','center_z','dim','rot','vel']
            HEAD_DICT: {{
                'center': {{'out_channels': 2, 'num_conv': 2}},
                'center_z': {{'out_channels': 1, 'num_conv': 2}},
                'dim': {{'out_channels': 3, 'num_conv': 2}},
                'rot': {{'out_channels': 2, 'num_conv': 2}},
                'vel': {{'out_channels': 2, 'num_conv': 2}},
            }}
        TARGET_ASSIGNER_CONFIG:
            FEATURE_MAP_STRIDE: 8
            NUM_MAX_OBJS: 500
            GAUSSIAN_OVERLAP: 0.1
            MIN_RADIUS: 2
        LOSS_CONFIG:
            LOSS_WEIGHTS: {{'cls_weight': 1.0, 'loc_weight': 0.25,
                            'code_weights': [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]}}
        POST_PROCESSING:
            SCORE_THRESH: 0.1
            POST_CENTER_LIMIT_RANGE: [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
            MAX_OBJ_PER_SAMPLE: 500
            NMS_CONFIG:
                NMS_TYPE: nms_gpu
                NMS_THRESH: 0.2
                NMS_PRE_MAXSIZE: 1000
                NMS_POST_MAXSIZE: 83
    POST_PROCESSING:
        RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
        EVAL_METRIC: kitti

OPTIMIZATION:
    BATCH_SIZE_PER_GPU: {batch_size}
    NUM_EPOCHS: {epochs}
    OPTIMIZER: adam_onecycle
    LR: {max_lr}
    WEIGHT_DECAY: 0.01
    MOMENTUM: 0.9
    MOMS: [0.95, 0.85]
    PCT_START: 0.4
    DIV_FACTOR: 10
    DECAY_STEP_LIST: [35, 45]
    LR_DECAY: 0.1
    LR_CLIP: 0.0000001
    LR_WARMUP: False
    WARMUP_EPOCH: 1
    GRAD_NORM_CLIP: 10
"""


# ----------------------------------------------------------------------
def prepare_kitti(cfg, pseudo_root):
    """
    Map raw-drive pseudo labels onto the object benchmark and rebuild infos.

    Uses devkit_object mapping files:
        train_mapping.txt:  <date> <drive> <raw_frame_idx>   (one per object idx)
        train_rand.txt:     permutation linking object idx -> mapping line
    """
    d = cfg["dataset"]
    mapping_dir = d["mapping_dir"]
    with open(os.path.join(mapping_dir, "train_rand.txt")) as f:
        rand = [int(x) for x in f.read().replace(",", " ").split()]
    with open(os.path.join(mapping_dir, "train_mapping.txt")) as f:
        mapping = [line.split() for line in f.read().strip().split("\n")]

    label_out = os.path.join(d["object_root"], "training", "label_pseudo")
    os.makedirs(label_out, exist_ok=True)
    n_written = 0
    for obj_idx, r in enumerate(rand):
        date, drive, raw_idx = mapping[r - 1]
        src = os.path.join(pseudo_root, drive, f"{int(raw_idx):010d}.txt")
        dst = os.path.join(label_out, f"{obj_idx:06d}.txt")
        if os.path.exists(src):
            shutil.copy(src, dst)
            n_written += 1
        else:
            open(dst, "w").close()                       # empty label file
    print(f"[KITTI] wrote {n_written} pseudo-label files to {label_out}")
    print("[KITTI] now point OpenPCDet's kitti_dataset.yaml at 'label_pseudo' "
          "(rename or symlink it to label_2 in a COPY of the dataset root) and run:\n"
          "  python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos "
          "tools/cfgs/dataset_configs/kitti_dataset.yaml")


def prepare_nuscenes(cfg, pseudo_root):
    """
    Inject pseudo-labels into OpenPCDet's nuscenes train infos
    (matched by LIDAR_TOP sample_data token) and subsample to 50%.
    Returns the filename of the new info pkl.
    """
    pcdet_root = cfg["self_training"]["openpcdet_root"]
    info_path = os.path.join(pcdet_root, "data", "nuscenes",
                             cfg["dataset"]["version"],
                             "nuscenes_infos_10sweeps_train.pkl")
    with open(info_path, "rb") as f:
        infos = pickle.load(f)

    # collect all pseudo infos keyed by lidar token
    pseudo = {}
    import glob
    for pkl in glob.glob(os.path.join(pseudo_root, "*_infos.pkl")):
        with open(pkl, "rb") as f:
            for e in pickle.load(f):
                pseudo[e["lidar_token"]] = e

    matched = 0
    new_infos = []
    for info in infos:
        tok = info.get("lidar_token") or os.path.basename(info["lidar_path"]).split(".")[0]
        if tok in pseudo:
            e = pseudo[tok]
            gt = e["gt_boxes"]
            # OpenPCDet nuscenes gt_boxes are 9-dim (x,y,z,l,w,h,yaw,vx,vy)
            info["gt_boxes"] = np.concatenate(
                [gt, np.zeros((len(gt), 2), np.float32)], axis=1)
            info["gt_names"] = e["gt_names"]
            info["num_lidar_pts"] = np.full(len(gt), 100)  # bypass min-pts filter
            matched += 1
            new_infos.append(info)
    print(f"[nuScenes] matched pseudo-labels for {matched}/{len(infos)} keyframes")

    # 50% training subsample, fixed seed (paper protocol)
    random.Random(42).shuffle(new_infos)
    new_infos = new_infos[: int(len(new_infos) * cfg["self_training"]["train_fraction"])]

    out_name = "nuscenes_infos_10sweeps_train_vilgod.pkl"
    out_path = os.path.join(os.path.dirname(info_path), out_name)
    with open(out_path, "wb") as f:
        pickle.dump(new_infos, f)
    print(f"[nuScenes] wrote {len(new_infos)} train infos to {out_path}")
    return out_name


def subsample_kitti_split(cfg):
    """Write a 50% train ImageSet split for the paper's protocol."""
    d = cfg["dataset"]
    split_file = os.path.join(d["object_root"], "ImageSets", "train.txt")
    with open(split_file) as f:
        ids = f.read().split()
    random.Random(42).shuffle(ids)
    ids = sorted(ids[: int(len(ids) * cfg["self_training"]["train_fraction"])])
    out = os.path.join(d["object_root"], "ImageSets", "train_vilgod.txt")
    with open(out, "w") as f:
        f.write("\n".join(ids))
    print(f"[KITTI] 50% split ({len(ids)} samples) -> {out} "
          "(set DATA_SPLIT train: train_vilgod in kitti_dataset.yaml)")


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pseudo-root", default=None,
                        help="defaults to <output_dir>/pseudo_labels")
    parser.add_argument("--launch", action="store_true",
                        help="actually launch OpenPCDet training")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    st = cfg["self_training"]
    pcdet = st["openpcdet_root"]
    pseudo_root = args.pseudo_root or os.path.join(
        cfg.get("output_dir", "outputs/" + cfg["dataset"]["name"]), "pseudo_labels")

    is_kitti = cfg["dataset"]["name"].lower() == "kitti"
    if is_kitti:
        prepare_kitti(cfg, pseudo_root)
        subsample_kitti_split(cfg)
        model_cfg = CENTERPOINT_KITTI_CFG.format(
            pcdet=pcdet, epochs=st["epochs"], max_lr=st["max_lr"],
            batch_size=st["batch_size_per_gpu"])
        cfg_name = "centerpoint_kitti_vilgod.yaml"
    else:
        train_info = prepare_nuscenes(cfg, pseudo_root)
        model_cfg = CENTERPOINT_NUSC_CFG.format(
            pcdet=pcdet, epochs=st["epochs"], max_lr=st["max_lr"],
            batch_size=st["batch_size_per_gpu"], train_info=repr(train_info))
        cfg_name = "centerpoint_nuscenes_vilgod.yaml"

    cfg_path = os.path.join(pcdet, "tools", "cfgs", cfg_name)
    with open(cfg_path, "w") as f:
        f.write(model_cfg)
    print(f"Wrote OpenPCDet config: {cfg_path}")

    cmd = [sys.executable, "train.py", "--cfg_file", f"cfgs/{cfg_name}"]
    print("Launch command:\n  cd", os.path.join(pcdet, "tools"), "&&", " ".join(cmd))
    if args.launch:
        subprocess.run(cmd, cwd=os.path.join(pcdet, "tools"), check=True)


if __name__ == "__main__":
    main()
