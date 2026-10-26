"""
Stage 5 check: Multi-target tracking and motion classification.

Runs GreedyTracker over a LiDAR sequence:
- Associates detections frame-by-frame using strict (1m) and relaxed (5m) radius matching.
- Classifies tracks into STATIC and MOVING based on point persistence and displacement.
- Plots track trajectories, bounding box footprints, and ego motion in the shared global frame.
"""
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

from data.nuscenes_loader import NuScenesSequenceLoader
from pipeline.ground_removal import GroundRemover
from pipeline.motion_score import PPScoreComputer
from pipeline.clustering import SpatioTemporalClusterer
from pipeline.bbox_fitting import BoxFitter
from pipeline.tracking import GreedyTracker

configuration = yaml.safe_load(open("config/nuscenes.yaml"))
sequence_loader = NuScenesSequenceLoader(configuration)
sequence = next(iter(sequence_loader))
frames = sequence["frames"][:20]
num_frames = len(frames)
print(f"sequence: {sequence['name']}, loaded {num_frames} frames")

ground_remover = GroundRemover(configuration["ground_removal"])
motion_score_computer = PPScoreComputer(configuration["motion_score"])
clusterer = SpatioTemporalClusterer(configuration["clustering"])
box_fitter = BoxFitter(configuration["bbox_fitting"])
tracker = GreedyTracker(configuration["tracking"])

# 1. Ground removal & global frame transform
nonground_global = []
ground_planes = []

for frame in frames:
    ng_pts, plane = ground_remover.remove(frame["points"][:, :3])
    ground_planes.append(plane)

    R = frame["pose"][:3, :3].astype(np.float32)
    t = frame["pose"][:3, 3].astype(np.float32)
    pts_global = np.dot(ng_pts, R.T) + t
    nonground_global.append(pts_global)

# 2. Motion scoring
pp_scores = motion_score_computer.compute_sequence(nonground_global)

# 3. Spatio-temporal clustering & box fitting across all frames
print("Clustering and fitting boxes across frames...")
detections = {}
n_agg = configuration["clustering"]["num_agg_frames"]

for t in range(num_frames):
    window = list(range(t, min(t + n_agg, num_frames)))
    segments = clusterer.cluster(
        frames_global=[nonground_global[j] for j in window],
        frames_pp=[pp_scores[j] for j in window],
        timestamps=[frames[j]["timestamp"] for j in window],
        ref_idx=0,
        ground_plane_sensor=ground_planes[t],
        ref_pose=frames[t]["pose"],
    )
    frame_dets = []
    for seg in segments:
        det = box_fitter.fit(seg)
        if det is not None:
            det.frame_idx = t
            frame_dets.append(det)
    detections[t] = frame_dets

total_raw_dets = sum(len(d) for d in detections.values())
print(f"Generated {total_raw_dets} total detections across {num_frames} frames (avg {total_raw_dets/num_frames:.1f}/frame)")

# 4. Multi-target tracking
print("Running GreedyTracker...")
tracks = tracker.run(detections, num_frames=num_frames)

static_tracks = [tr for tr in tracks if tr.is_static]
moving_tracks = [tr for tr in tracks if not tr.is_static]
long_tracks = [tr for tr in tracks if len(tr.detections) >= 3]
long_moving = [tr for tr in moving_tracks if len(tr.detections) >= 3]

print(f"\nTracking Results Summary:")
print(f"  - Total tracks formed: {len(tracks)}")
print(f"  - Static tracks: {len(static_tracks)} ({100*len(static_tracks)/len(tracks):.1f}%)")
print(f"  - Moving tracks: {len(moving_tracks)} ({100*len(moving_tracks)/len(tracks):.1f}%)")
print(f"  - Tracks surviving >= 3 frames: {len(long_tracks)}")
print(f"  - Moving tracks surviving >= 3 frames: {len(long_moving)}")

for i, tr in enumerate(long_moving[:5]):
    c0 = tr.detections[0].box[:2]
    c1 = tr.detections[-1].box[:2]
    disp = np.linalg.norm(c1 - c0)
    dur = tr.detections[-1].frame_idx - tr.detections[0].frame_idx + 1
    print(f"    Moving Track ID {tr.id:3d}: {len(tr.detections)} detections over {dur} frames, displacement = {disp:.2f} m")

# Helper for 2D box corners in GLOBAL frame
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

# 5. Visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Ego trajectory in global frame
ego_xy = np.stack([f["pose"][:2, 3] for f in frames])

# Panel 1: Global Trajectory Map
ax1 = axes[0]
ax1.plot(ego_xy[:, 0], ego_xy[:, 1], "k-o", linewidth=2.5, markersize=5, label="Ego vehicle path")
ax1.scatter(ego_xy[0, 0], ego_xy[0, 1], c="green", s=100, marker="^", label="Ego start")
ax1.scatter(ego_xy[-1, 0], ego_xy[-1, 1], c="black", s=100, marker="s", label="Ego end")

# Plot static track anchor positions
static_pts = np.array([tr.detections[0].box[:2] for tr in static_tracks if len(tr.detections) >= 3])
if len(static_pts) > 0:
    ax1.scatter(static_pts[:, 0], static_pts[:, 1], c="cornflowerblue", s=25, alpha=0.7, label="Static tracks (>=3 frames)")

# Plot moving trajectories
cmap = plt.colormaps["tab10"]
for i, tr in enumerate(moving_tracks):
    if len(tr.detections) >= 2:
        traj = np.array([d.box[:2] for d in tr.detections])
        color = cmap(i % 10)
        ax1.plot(traj[:, 0], traj[:, 1], "-o", color=color, linewidth=2.2, markersize=4, alpha=0.9)
        # label track ID
        ax1.text(traj[-1, 0] + 0.5, traj[-1, 1] + 0.5, f"ID {tr.id}", fontsize=8, color=color, weight="bold")

ax1.set_title(f"Stage 5: Global Multi-Target Tracking ({len(tracks)} Tracks, {len(moving_tracks)} Moving)")
ax1.set_xlabel("Global X (m)")
ax1.set_ylabel("Global Y (m)")
ax1.axis("equal")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle=":", alpha=0.6)

# Panel 2: Detailed Zoomed View of Moving Tracks with Bounding Boxes
ax2 = axes[1]
# Draw non-ground points from middle frame for spatial context
mid_t = num_frames // 2
mid_pts = nonground_global[mid_t]
ax2.scatter(mid_pts[:, 0], mid_pts[:, 1], s=0.3, c="lightgray", alpha=0.6, label=f"LiDAR points (t={mid_t})")

# Draw ego vehicle path
ax2.plot(ego_xy[:, 0], ego_xy[:, 1], "k--", linewidth=1.5, alpha=0.7, label="Ego path")

# Draw footprints of long tracks
for i, tr in enumerate(long_tracks):
    is_m = not tr.is_static
    if not is_m and len(tr.detections) < 5:
        continue  # skip short static tracks for clarity
    
    color = "crimson" if is_m else "navy"
    alpha = 0.8 if is_m else 0.3
    
    # Plot trajectory line
    traj = np.array([d.box[:2] for d in tr.detections])
    ax2.plot(traj[:, 0], traj[:, 1], "-", color=color, linewidth=2.0 if is_m else 1.0, alpha=alpha)
    
    # Draw box for first and last detection
    for d_idx in [0, len(tr.detections) - 1]:
        d = tr.detections[d_idx]
        corners = get_box_2d_corners(d.box)
        poly = Polygon(corners, closed=True, fill=False, edgecolor=color,
                       linewidth=1.8 if is_m else 1.0, linestyle="-" if is_m else ":", alpha=alpha)
        ax2.add_patch(poly)

legend_elements = [
    Line2D([0], [0], color="crimson", lw=2.0, label="Moving Track (Trajectories & Boxes)"),
    Line2D([0], [0], color="navy", lw=1.0, ls=":", label="Static Track Footprint"),
    Line2D([0], [0], color="black", lw=1.5, ls="--", label="Ego Path"),
]
ax2.legend(handles=legend_elements, loc="upper right")
ax2.set_title("Track Footprints & Motion Paths (Red = Moving, Blue = Static)")
ax2.set_xlabel("Global X (m)")
ax2.set_ylabel("Global Y (m)")

# Focus view on active region
ax2.set_xlim(ego_xy[:, 0].mean() - 40, ego_xy[:, 0].mean() + 40)
ax2.set_ylim(ego_xy[:, 1].mean() - 40, ego_xy[:, 1].mean() + 40)
ax2.axis("equal")
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("check_05_tracking.png", dpi=140)
plt.close()
print("saved check_05_tracking.png")

