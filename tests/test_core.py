import numpy as np
from pipeline.motion_score import PPScoreComputer

def test_static_vs_moving():
    pp = PPScoreComputer({'radius': 0.6, 'num_adjacent': 3})
    # a pile of points that never moves (a "wall")
    static = np.tile([[5., 5., 0.]], (30, 1)) + 0.05*np.random.randn(30, 3)
    frames = []
    for t in range(7):
        # a pile of points that slides sideways each frame (a "car")
        moving = np.tile([[t*3., 0., 0.]], (30, 1)) + 0.05*np.random.randn(30, 3)
        frames.append(np.vstack([static, moving]).astype(np.float32))
    scores = pp.compute_sequence(frames)
    assert scores[3][:30].mean() > 0.9   # the wall should score HIGH (=still)
    assert scores[3][30:].mean() < 0.3   # the car should score LOW  (=moving)