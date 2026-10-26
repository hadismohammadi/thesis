import numpy as np
import yaml
import matplotlib.pyplot as plt
from data.nuscenes_loader import NuScenesSequenceLoader

cfg = yaml.safe_load(open("config/nuscenes.yaml"))
loader = NuScenesSequenceLoader(cfg)
seq = next(iter(loader))
frames = seq["frames"][:5]

# same transform as your test: move each scan into the shared map frame
clouds = []
for fr in frames:
    pts = fr["points"][:, :3]
    pts = pts[np.isfinite(pts).all(axis=1)]      # drop broken rows (see previous fix)
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    clouds.append((fr["pose"] @ hom.T).T[:, :3])

# remove ground points (roughly bottom 1m) so buildings/poles stand out clearly
def drop_ground(c):
    return c[c[:, 2] > (c[:, 2].min() + 1.0)]

c0 = drop_ground(clouds[0])   # first scan
c4 = drop_ground(clouds[4])   # a later scan (5 scans later)

plt.figure(figsize=(10, 10))
plt.scatter(c0[:, 0], c0[:, 1], s=0.5, c="red",  alpha=0.5, label="scan 0 (first)")
plt.scatter(c4[:, 0], c4[:, 1], s=0.5, c="blue", alpha=0.5, label="scan 4 (later)")
plt.legend(markerscale=15)
plt.title("Top-down view: do the two scans line up?")
plt.xlabel("x (meters)")
plt.ylabel("y (meters)")
plt.axis("equal")
plt.savefig("alignment_check.png", dpi=150)
plt.show()
print("Saved plot to alignment_check.png")