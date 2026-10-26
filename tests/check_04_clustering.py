"""
Stage 4 check: Spatio-temporal clustering and bounding box fitting.

Aggregates a temporal window of non-ground frames, clusters object proposals
using HDBSCAN on [x, y, z, w_pp * pp, w_dt * dt] feature vectors, fits oriented
3D bounding boxes with L-shape fitting, and compares against ground-truth objects.
"""
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from data.nuscenes_loader import NuScenesSequenceLoader
from pipeline.ground_removal import GroundRemover
from pipeline.motion_score import PPScoreComputer
from pipeline.clustering import SpatioTemporalClusterer
from pipeline.bbox_fitting import BoxFitter
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import points_in_box

configuration = yaml.safe_load(open("config/nuscenes.yaml"))
sequence_loader = NuScenesSequenceLoader(configuration)
sequence = next(iter(sequence_loader))
frames = sequence["frames"][:15]
print(f"sequence: {sequence['name']}, loaded {len(frames)} frames")

ground_remover = GroundRemover(configuration["ground_removal"])
motion_score_computer = PPScoreComputer(configuration["motion_score"])
clusterer = SpatioTemporalClusterer(configuration["clustering"])
box_fitter = BoxFitter(configuration["bbox_fitting"])

# Process non-ground points and transformations
nonground_sensor = []
nonground_global = []
ground_planes = []

for frame in frames:
    ng_pts, plane = ground_remover.remove(frame["points"][:, :3])
    nonground_sensor.append(ng_pts)
    ground_planes.append(plane)

    R = frame["pose"][:3, :3].astype(np.float32)
    t = frame["pose"][:3, 3].astype(np.float32)
    pts_global = np.dot(ng_pts, R.T) + t
    nonground_global.append(pts_global)

pp_scores = motion_score_computer.compute_sequence(nonground_global)

# Use reference frame (middle frame 7) for full temporal aggregation context
ref_idx = len(frames) // 2
n_agg = configuration["clustering"]["num_agg_frames"]
window = list(range(ref_idx, min(ref_idx + n_agg, len(frames))))
ref_frame = frames[ref_idx]

print(f"\nClustering reference frame {ref_idx} with {len(window)} aggregated frames...")
segments = clusterer.cluster(
    frames_global=[nonground_global[j] for j in window],
    frames_pp=[pp_scores[j] for j in window],
    timestamps=[frames[j]["timestamp"] for j in window],
    ref_idx=0,
    ground_plane_sensor=ground_planes[ref_idx],
    ref_pose=ref_frame["pose"],
)

print(f"HDBSCAN clustering produced {len(segments)} valid object segments (post-filter)")

# Fit 3D bounding boxes
detections = []
for seg in segments:
    det = box_fitter.fit(seg)
    if det is not None:
        det.frame_idx = ref_idx
        det.is_moving = (seg.pp_scores.mean() < configuration["clustering"]["motion_keep_threshold"])
        detections.append(det)

print(f"Fitted {len(detections)} 3D bounding boxes")
num_moving = sum(1 for d in detections if d.is_moving)
num_static = len(detections) - num_moving
print(f"  - Moving proposals (mean PP < 0.5): {num_moving}")
print(f"  - Static proposals (mean PP >= 0.5): {num_static}")

# Transform fitted boxes into reference sensor frame for plotting
T_gs = np.linalg.inv(ref_frame["pose"])
R_gs = T_gs[:3, :3]
t_gs = T_gs[:3, 3]

def box_global_to_sensor(box_g):
    cx, cy, cz, l, w, h, yaw = box_g
    center_s = np.dot(R_gs, np.array([cx, cy, cz])) + t_gs
    # rotate yaw by sensor heading change
    yaw_rot = np.arctan2(R_gs[1, 0] * np.cos(yaw) + R_gs[1, 1] * np.sin(yaw),
                         R_gs[0, 0] * np.cos(yaw) + R_gs[0, 1] * np.sin(yaw))
    return np.array([center_s[0], center_s[1], center_s[2], l, w, h, yaw_rot])

def get_box_2d_corners(box):
    cx, cy, _, l, w, _, yaw = box
    cos, sin = np.cos(yaw), np.sin(yaw)
    dx, dy = l / 2.0, w / 2.0
    corners = np.array([
        [cx + dx * cos - dy * sin, cy + dx * sin + dy * cos],
        [cx - dx * cos - dy * sin, cy - dx * sin + dy * cos],
        [cx - dx * cos + dy * sin, cy - dx * sin - dy * cos],
        [cx + dx * cos + dy * sin, cy + dx * sin - dy * cos],
    ])
    return corners

# Load ground-truth boxes in reference frame
nusc = sequence_loader.nusc
sample_data = nusc.get("sample_data", ref_frame["token"])
_, gt_boxes, _ = nusc.get_sample_data(sample_data["token"])
cs = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
q_sensor = Quaternion(cs["rotation"])
t_sensor = np.array(cs["translation"])
for b in gt_boxes:
    b.rotate(q_sensor)
    b.translate(t_sensor)

# Plotting: Left = Clusters & Proposed Boxes, Right = Comparison with GT
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Panel 1: Cluster segments and fitted boxes
ax1 = axes[0]
ref_pts = nonground_sensor[ref_idx]
ax1.scatter(ref_pts[:, 0], ref_pts[:, 1], s=0.5, c="gainsboro", label="background / unclustered")

cmap = plt.colormaps["tab20"]
for i, det in enumerate(detections):
    seg_pts = det.segment.points_sensor
    color = cmap(i % 20)
    ax1.scatter(seg_pts[:, 0], seg_pts[:, 1], s=2.0, color=color)

    # Draw fitted box
    box_s = box_global_to_sensor(det.box)
    corners = get_box_2d_corners(box_s)
    poly = Polygon(corners, closed=True, fill=False, edgecolor="crimson" if det.is_moving else "blue",
                   linewidth=1.8, linestyle="-" if det.is_moving else "--")
    ax1.add_patch(poly)

ax1.scatter([0], [0], c="black", s=140, marker="^", label="ego vehicle")
ax1.set_title(f"Stage 4: {len(detections)} Fitted Boxes (Red = Moving, Blue = Static)")
ax1.set_xlabel("x forward (m)")
ax1.set_ylabel("y left (m)")
ax1.set_xlim(-50, 50)
ax1.set_ylim(-50, 50)
ax1.axis("equal")
ax1.legend(loc="upper right")

# Panel 2: Comparison against nuScenes Ground Truth Objects
ax2 = axes[1]
ax2.scatter(ref_pts[:, 0], ref_pts[:, 1], s=0.5, c="lightgray", label="LiDAR points")

# Plot GT boxes
for b in gt_boxes:
    c = b.center
    l, w, h = b.wlh[1], b.wlh[0], b.wlh[2]  # nuScenes wlh: [width, length, height]
    yaw = b.orientation.yaw_pitch_roll[0]
    gt_box_arr = np.array([c[0], c[1], c[2], l, w, h, yaw])
    corners = get_box_2d_corners(gt_box_arr)
    poly = Polygon(corners, closed=True, fill=False, edgecolor="green", linewidth=1.5, linestyle=":")
    ax2.add_patch(poly)

# Overlay fitted proposal boxes
for det in detections:
    box_s = box_global_to_sensor(det.box)
    corners = get_box_2d_corners(box_s)
    poly = Polygon(corners, closed=True, fill=False, edgecolor="crimson" if det.is_moving else "blue",
                   linewidth=1.8, alpha=0.85)
    ax2.add_patch(poly)

# Legend handles
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="green", lw=1.5, ls=":", label="Ground Truth (nuScenes)"),
    Line2D([0], [0], color="crimson", lw=1.8, label="Fitted Moving Proposal"),
    Line2D([0], [0], color="blue", lw=1.8, label="Fitted Static Proposal"),
    Line2D([0], [0], marker="^", color="black", lw=0, markersize=8, label="Ego vehicle"),
]
ax2.legend(handles=legend_elements, loc="upper right")
ax2.set_title("Proposals (Red/Blue) vs Ground Truth (Green)")
ax2.set_xlabel("x forward (m)")
ax2.set_ylabel("y left (m)")
ax2.set_xlim(-50, 50)
ax2.set_ylim(-50, 50)
ax2.axis("equal")

plt.tight_layout()
plt.savefig("check_04_clustering.png", dpi=140)
plt.close()
print("saved check_04_clustering.png")
