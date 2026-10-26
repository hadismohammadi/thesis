"""
Stage 3 check: motion scoring (PP-score).

Assigns every non-ground point a persistence score between 0 and 1, then draws
the scene coloured by that score. Static structure should be blue, moving
objects should stand out as red clusters.
"""
import numpy as np
import yaml
import matplotlib.pyplot as plt

from data.nuscenes_loader import NuScenesSequenceLoader
from pipeline.ground_removal import GroundRemover
from pipeline.motion_score import PPScoreComputer

configuration = yaml.safe_load(open("config/nuscenes.yaml"))
sequence_loader = NuScenesSequenceLoader(configuration)
sequence = next(iter(sequence_loader))
frames = sequence["frames"][:15]
print(f"sequence: {sequence['name']}, loaded {len(frames)} frames")

ground_remover = GroundRemover(configuration["ground_removal"])

# The motion score compares each point against neighbouring frames, so every
# frame must be expressed in the same shared map coordinates.
non_ground_in_sensor_frame = []
non_ground_in_global_frame = []
for frame in frames:
    non_ground_points, _ = ground_remover.remove(frame["points"][:, :3])
    non_ground_in_sensor_frame.append(non_ground_points)

    homogeneous_points = np.hstack([non_ground_points,
                                    np.ones((len(non_ground_points), 1))])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        points_in_global_frame = (frame["pose"] @ homogeneous_points.T).T[:, :3]
    non_ground_in_global_frame.append(points_in_global_frame.astype(np.float32))

motion_score_computer = PPScoreComputer(configuration["motion_score"])
motion_scores_per_frame = motion_score_computer.compute_sequence(non_ground_in_global_frame)

# Use a middle frame: it has neighbouring frames on both sides, so its scores
# are computed with full temporal context.
middle_frame_index = len(frames) // 2
motion_scores = motion_scores_per_frame[middle_frame_index]
points_to_plot = non_ground_in_sensor_frame[middle_frame_index]

fraction_moving = (motion_scores < 0.5).mean()
fraction_static = (motion_scores > 0.7).mean()
print(f"\nframe {middle_frame_index}: {len(motion_scores)} non-ground points")
print(f"  mean score   = {motion_scores.mean():.3f}")
print(f"  median score = {np.median(motion_scores):.3f}")
print(f"  looks moving (score < 0.5): {100 * fraction_moving:.1f}% of points")
print(f"  looks static (score > 0.7): {100 * fraction_static:.1f}% of points")

figure, axis = plt.subplots(figsize=(11, 10))
scatter = axis.scatter(points_to_plot[:, 0], points_to_plot[:, 1],
                       s=1.2, c=motion_scores, cmap="coolwarm_r", vmin=0, vmax=1)
plt.colorbar(scatter, label="persistence score (0 = moving, 1 = static)")
axis.scatter([0], [0], c="black", s=120, marker="^", label="ego vehicle")
axis.set_title(f"Motion score, frame {middle_frame_index} — "
               f"red clusters should be moving objects")
axis.set_xlabel("x forward (m)")
axis.set_ylabel("y left (m)")
axis.axis("equal")
axis.legend()

plt.tight_layout()
plt.savefig("check_03_motion_score.png", dpi=140)
plt.close()
print("saved check_03_motion_score.png")