import numpy as np, yaml
from data.nuscenes_loader import NuScenesSequenceLoader

def test_frames_line_up():
    config = yaml.safe_load(open("config/nuscenes.yaml"))
    loader = NuScenesSequenceLoader(config)
    sequence = next(iter(loader))
    frames = sequence["frames"][:20]

    # for frame_index, frame in enumerate(frames):
    #     pose = frame["pose"]
    #     print(f"frame {frame_index}: pose finite = {np.isfinite(pose).all()}")
    #     if not np.isfinite(pose).all():
    #         print(pose)

    # for frame_index, frame in enumerate(frames):
    #     points = frame["points"][:, :3]
    #     homogeneous_points = np.hstack([points, np.ones((len(points), 1))])
    #     with np.errstate(all="raise"):     # turn the warning into a hard stop, right here
    #         try:
    #             result = frame["pose"] @ homogeneous_points.T
    #         except FloatingPointError as e:
    #             print(f"frame {frame_index}: BROKEN — {e}")
    #             print("pose:\n", frame["pose"])
    #             break

    # first_frame = frames[0]
    # first_frame_points = first_frame["points"][:, :3]
    # first_frame_homogeneous_points = np.hstack([first_frame_points, np.ones((len(first_frame_points), 1))])
    # result = first_frame["pose"] @ first_frame_homogeneous_points.T          # let the warning print, don't stop this time
    # print("output contains any bad values?", not np.isfinite(result).all())

    global_point_clouds = []
    for frame in frames:
        points = frame["points"][:, :3]
        homogeneous_points = np.hstack([points, np.ones((len(points), 1))])
        global_point_clouds.append((frame["pose"] @ homogeneous_points.T).T[:, :3])

    def to_occupancy_grid(points, map_origin, cell_size_meters=1.0, half_width_meters=50):
        grid_indices = np.floor((points[:, :2] - map_origin + half_width_meters) / cell_size_meters).astype(int)
        in_bounds_mask = (grid_indices >= 0).all(1) & (grid_indices < int(2*half_width_meters/cell_size_meters)).all(1)
        occupancy_grid = np.zeros((int(2*half_width_meters/cell_size_meters),)*2, bool)
        occupancy_grid[grid_indices[in_bounds_mask, 0], grid_indices[in_bounds_mask, 1]] = True
        return occupancy_grid

    neighbor_overlaps = []
    for frame_index in range(len(global_point_clouds) - 1):
        map_origin = global_point_clouds[frame_index][:, :2].mean(axis=0)
        current_frame_grid = to_occupancy_grid(global_point_clouds[frame_index], map_origin)
        next_frame_grid = to_occupancy_grid(global_point_clouds[frame_index + 1], map_origin)
        overlap = (current_frame_grid & next_frame_grid).sum() / (current_frame_grid | next_frame_grid).sum()
        neighbor_overlaps.append(overlap)
        print(f"frame {frame_index} -> {frame_index+1}: overlap = {round(float(overlap), 3)}")

    neighbor_overlaps = np.array(neighbor_overlaps)
    print("mean neighbor overlap:", round(float(neighbor_overlaps.mean()), 3))
    assert neighbor_overlaps.mean() > 0.6          # neighbors should agree strongly
    assert neighbor_overlaps.min() > 0.4           # no single pair should collapse
