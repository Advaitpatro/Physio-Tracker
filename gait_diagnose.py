#!/usr/bin/env python3
"""
gait_diagnose.py
----------------
Reads a real session JSON and prints the numbers needed to fix the gait
thresholds (why steps / cadence / step length come out wrong).

RUN:
    python gait_diagnose.py <session>.json
    (use a walking session you can also count by eye)
"""

import sys
import json
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "skeleton_live.json"
d = json.load(open(path))
names = d["jointNames"]
idx = {n: i for i, n in enumerate(names)}
frames = d["frames"]
t = np.array([f["t"] for f in frames], float)
n = len(frames)
fps = 1.0 / np.median(np.diff(t))


def joint(name):
    i = idx[name]
    a = np.array([[f["px"][i], f["py"][i], f["pz"][i]] for f in frames], float)
    zero = (a[:, 0] == 0) & (a[:, 1] == 0) & (a[:, 2] == 0)
    a[zero] = np.nan
    ii = np.arange(n)
    for c in range(3):
        col = a[:, c]; g = ~np.isnan(col)
        if g.sum() >= 2:
            a[:, c] = np.interp(ii, ii[g], col[g])
    return a


print(f"\nFile: {path}")
print(f"Frames: {n}   Duration: {t[-1]-t[0]:.1f}s   FPS: {fps:.1f}")

hip = (joint("LEFT_HIP") + joint("RIGHT_HIP")) / 2.0
hip_travel = np.linalg.norm(hip[-1, [0, 2]] - hip[0, [0, 2]])
hip_path = np.sum(np.linalg.norm(np.diff(hip[:, [0, 2]], axis=0), axis=1))
print(f"Hip straight-line travel: {hip_travel:.2f} m   |  total hip path: {hip_path:.2f} m")

print("\nPer-foot ground speed (m/s) -- this sets the stance threshold:")
for side in ("LEFT", "RIGHT"):
    g = joint(side + "_ANKLE")[:, [0, 2]]
    speed = np.r_[0, np.linalg.norm(np.diff(g, axis=0), axis=1)] * fps
    # light smoothing
    k = 5
    speed = np.convolve(speed, np.ones(k) / k, mode="same")
    print(f"  {side} ankle: min {speed.min():.2f}  median {np.median(speed):.2f}  "
          f"mean {speed.mean():.2f}  90th% {np.percentile(speed,90):.2f}  max {speed.max():.2f}")

print("\nVertical range of ankles (m) -- sanity check the feet were tracked:")
for side in ("LEFT", "RIGHT"):
    y = joint(side + "_ANKLE")[:, 1]
    print(f"  {side} ankle Y: min {y.min():.2f}  max {y.max():.2f}  range {y.max()-y.min():.2f}")

print("\nHow many frames each ankle sits BELOW various speed cutoffs")
print("(a good stance threshold makes stance ~40-60% of frames):")
for side in ("LEFT", "RIGHT"):
    g = joint(side + "_ANKLE")[:, [0, 2]]
    speed = np.convolve(np.r_[0, np.linalg.norm(np.diff(g, axis=0), axis=1)] * fps,
                        np.ones(5) / 5, mode="same")
    print(f"  {side}: ", end="")
    for thr in (0.05, 0.10, 0.18, 0.30, 0.50):
        pct = 100 * np.mean(speed < thr)
        print(f"<{thr}:{pct:.0f}%  ", end="")
    print()
print()
