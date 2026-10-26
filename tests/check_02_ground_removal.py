"""
Stage 2 check: ground removal.

Removes the road surface from a real LiDAR scan and draws the result, so we can
confirm by eye that the road is gone and the objects remain.
"""
import numpy as np
import yaml
import matplotlib.pyplot as plt

from data.nuscenes_loader import NuScenesSequenceLoader
from pipeline.ground_removal import GroundRemover

configuration = yaml.safe_load(open("config/nuscenes.yaml"))
sequence_loader = NuScenesSequenceLoader(configuration)
sequence = next(iter(sequence_loader))
frames = sequence["frames"][:15]
print(f"sequence: {sequence['name']}, loaded {len(frames)} frames")

ground_remover = GroundRemover(configuration["ground_removal"])

first_frame = frames[0]
all_points = first_frame["points"][:, :3]
non_ground_points, ground_plane = ground_remover.remove(all_points)

percent_kept = 100 * len(non_ground_points) / len(all_points)
print(f"frame 0: {len(all_points)} points -> {len(non_ground_points)} non-ground "
      f"({percent_kept:.0f}% kept)")
print(f"frame 0: ground plane coefficients = {np.round(ground_plane, 3)}")

# Mark which of the original points survived, so we can draw both groups.
non_ground_lookup = set(map(tuple, np.round(non_ground_points, 4)))
is_non_ground = np.array([tuple(point) in non_ground_lookup
                          for point in np.round(all_points, 4)])

figure, axes = plt.subplots(1, 2, figsize=(16, 7))

top_view = axes[0]
top_view.scatter(all_points[~is_non_ground, 0], all_points[~is_non_ground, 1],
                 s=0.3, c="lightgray", label="ground (removed)")
top_view.scatter(all_points[is_non_ground, 0], all_points[is_non_ground, 1],
                 s=0.3, c="crimson", label="non-ground (kept)")
top_view.set_title("Top view: ground versus non-ground")
top_view.set_xlabel("x forward (m)")
top_view.set_ylabel("y left (m)")
top_view.legend(markerscale=20)
top_view.axis("equal")

side_view = axes[1]
side_view.scatter(all_points[~is_non_ground, 0], all_points[~is_non_ground, 2],
                  s=0.3, c="lightgray")
side_view.scatter(all_points[is_non_ground, 0], all_points[is_non_ground, 2],
                  s=0.3, c="crimson")
side_view.set_title("Side view: was the road surface actually removed?")
side_view.set_xlabel("x forward (m)")
side_view.set_ylabel("z up (m)")
side_view.set_ylim(-3, 6)

plt.tight_layout()
plt.savefig("check_02_ground_removal.png", dpi=140)
plt.close()
print("saved check_02_ground_removal.png")