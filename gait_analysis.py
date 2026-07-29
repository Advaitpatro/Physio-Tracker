#!/usr/bin/env python3
"""
gait_analysis.py
----------------
Computes spatiotemporal gait parameters for a TUG session from the recorded 3D
joints (skeleton_live.json from the app, or a *_skeleton_live.json export).

Parameters produced:
  number of steps, cadence, step length, stride length, step time,
  step-to-step variability, step width, double-support time, turn time,
  steps during the turn, shuffling count, freezing episodes.

IMPORTANT (single-camera honesty): reliability is printed next to each value.
  HIGH/MODERATE  = usable from one depth camera
  LOW            = weak from a single view; treat as indicative only
                   (step width, shuffling, freezing especially)

SETUP:
    pip install numpy
RUN:
    python gait_analysis.py skeleton_live.json
"""

import sys
import json
import numpy as np

# --- tunable thresholds (metres, seconds, m/s) ---
STANCE_SPEED = 0.18    # foot ground-speed below this = planted (stance)
MIN_STANCE = 0.10      # ignore stance blips shorter than this
FREEZE_SPEED = 0.10    # hip forward speed below this = not progressing
FREEZE_DUR = 0.5       # a freeze must last at least this long
CLEAR_THRESH = 0.03    # swing foot clearance below this = shuffling
SHORT_STEP = 0.15      # step shorter than this contributes to shuffling flag


def load(path):
    d = json.load(open(path))
    names = d["jointNames"]
    idx = {n: i for i, n in enumerate(names)}
    frames = d["frames"]
    t = np.array([f["t"] for f in frames], float)

    def joint(name):
        i = idx[name]
        a = np.array([[f["px"][i], f["py"][i], f["pz"][i]] for f in frames], float)
        zero = (a[:, 0] == 0) & (a[:, 1] == 0) & (a[:, 2] == 0)   # app writes 0,0,0 when depth missing
        a[zero] = np.nan
        ii = np.arange(len(a))
        for c in range(3):
            col = a[:, c]; good = ~np.isnan(col)
            if good.sum() >= 2:
                a[:, c] = np.interp(ii, ii[good], col[good])
        return a

    return t, joint


def smooth(x, k=5):
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def runs(mask, min_len):
    """Return list of (start,end) index runs where mask is True and long enough."""
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i)); s = None
    if s is not None:
        out.append((s, len(mask)))
    return [(a, b) for a, b in out if b - a >= min_len]


def analyse(path):
    t, joint = load(path)
    n = len(t)
    if n < 10:
        print("Not enough frames to analyse."); return
    fps = 1.0 / np.median(np.diff(t))
    min_stance_f = max(2, int(MIN_STANCE * fps))

    # up axis = Y (index 1); ground plane = (X, Z) = indices 0, 2
    hip = (joint("LEFT_HIP") + joint("RIGHT_HIP")) / 2.0
    hip_g = hip[:, [0, 2]]
    # progression axis via PCA of the hip path on the ground
    c = hip_g - hip_g.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    u = vt[0]                      # forward/progression direction
    v = np.array([-u[1], u[0]])    # lateral direction

    def ap(pg):   # along walking direction
        return (pg - hip_g.mean(0)) @ u

    def ml(pg):   # side to side
        return (pg - hip_g.mean(0)) @ v

    # --- per-foot stance detection & footfalls ---
    foot = {}
    for side in ("LEFT", "RIGHT"):
        ank = joint(side + "_ANKLE")
        g = ank[:, [0, 2]]
        height = ank[:, 1]
        speed = smooth(np.r_[0, np.linalg.norm(np.diff(g, axis=0), axis=1)] * fps)
        stance = speed < STANCE_SPEED
        intervals = runs(stance, min_stance_f)
        falls = []
        for k, (a, b) in enumerate(intervals):
            mid = (a + b) // 2
            # swing clearance = how high the foot rose since the previous stance
            prev_end = intervals[k - 1][1] if k > 0 else 0
            clearance = float(np.nanmax(height[prev_end:a + 1]) - np.nanmin(height[a:b + 1])) if a > prev_end else 0.0
            falls.append({"t": t[a], "ap": float(ap(g[mid:mid + 1])[0]),
                          "ml": float(ml(g[mid:mid + 1])[0]), "clear": clearance, "side": side})
        foot[side] = {"stance": stance, "falls": falls}

    footfalls = sorted(foot["LEFT"]["falls"] + foot["RIGHT"]["falls"], key=lambda f: f["t"])
    n_steps = len(footfalls)

    # --- turn detection: farthest point along progression, heading reversal ---
    apc = smooth(ap(hip_g), 7)
    turn_i = int(np.argmax(apc))
    fwd_speed = smooth(np.r_[0, np.diff(apc)] * fps, 7)
    # turn window = around the apex where forward speed is near zero / negative
    lo = turn_i
    while lo > 0 and fwd_speed[lo] > -0.05 and (turn_i - lo) < fps * 2.5:
        lo -= 1
    hi = turn_i
    while hi < n - 1 and fwd_speed[hi] < 0.05 and (hi - turn_i) < fps * 2.5:
        hi += 1
    turn_time = t[hi] - t[lo]
    steps_in_turn = sum(1 for f in footfalls if t[lo] <= f["t"] <= t[hi])

    # --- step / stride / width / times ---
    step_lengths, step_times, step_widths = [], [], []
    for i in range(1, len(footfalls)):
        if footfalls[i]["side"] != footfalls[i - 1]["side"]:   # opposite feet = a step
            step_lengths.append(abs(footfalls[i]["ap"] - footfalls[i - 1]["ap"]))
            step_times.append(footfalls[i]["t"] - footfalls[i - 1]["t"])
            step_widths.append(abs(footfalls[i]["ml"] - footfalls[i - 1]["ml"]))
    stride_lengths = []
    for side in ("LEFT", "RIGHT"):
        fs = [f for f in footfalls if f["side"] == side]
        stride_lengths += [abs(fs[i]["ap"] - fs[i - 1]["ap"]) for i in range(1, len(fs))]

    walk_dur = t[-1] - t[0]
    cadence = 60.0 * n_steps / walk_dur if walk_dur > 0 else float("nan")

    def mean(x):
        return float(np.mean(x)) if x else float("nan")

    def cv(x):
        return float(np.std(x) / np.mean(x) * 100) if x and np.mean(x) else float("nan")

    # --- double support: both feet in stance at once ---
    both = foot["LEFT"]["stance"] & foot["RIGHT"]["stance"]
    double_support_t = float(np.sum(both) / fps)
    double_support_pct = float(100.0 * np.sum(both) / n)

    # --- shuffling: footfalls with low clearance or very short step ---
    shuffles = sum(1 for f in footfalls if f["clear"] < CLEAR_THRESH)
    shuffles += sum(1 for s in step_lengths if s < SHORT_STEP)

    # --- freezing (experimental proxy): non-progression episodes outside the turn ---
    hipspeed = smooth(np.r_[0, np.linalg.norm(np.diff(hip_g, axis=0), axis=1)] * fps, 7)
    frozen = hipspeed < FREEZE_SPEED
    frozen[max(0, lo):min(n, hi)] = False        # ignore the turn/standing
    freeze_runs = runs(frozen, int(FREEZE_DUR * fps))
    freeze_episodes = len(freeze_runs)
    freeze_time = float(sum(t[min(b, n - 1)] - t[a] for a, b in freeze_runs))

    # ---------- report ----------
    def line(label, val, unit, rel, fmt="{:.2f}"):
        s = "  n/a" if (isinstance(val, float) and val != val) else fmt.format(val)
        print(f"  {label:<26}{s:>8} {unit:<5} [{rel}]")

    print("\n" + "=" * 60)
    print("  SPATIOTEMPORAL GAIT SUMMARY")
    print(f"  duration {walk_dur:.1f}s   ~{fps:.0f} fps")
    print("=" * 60)
    line("Number of steps", n_steps, "", "MODERATE", "{:.0f}")
    line("Cadence", cadence, "spm", "MODERATE", "{:.0f}")
    line("Step length (mean)", mean(step_lengths), "m", "MODERATE")
    line("Stride length (mean)", mean(stride_lengths), "m", "MODERATE")
    line("Step time (mean)", mean(step_times), "s", "MODERATE")
    line("Step-length variability", cv(step_lengths), "%CV", "MODERATE", "{:.1f}")
    line("Step-time variability", cv(step_times), "%CV", "MODERATE", "{:.1f}")
    line("Double support time", double_support_t, "s", "MODERATE")
    line("Double support", double_support_pct, "%", "MODERATE", "{:.0f}")
    line("Turn time", turn_time, "s", "MODERATE")
    line("Steps during turn", steps_in_turn, "", "MODERATE", "{:.0f}")
    print("  " + "-" * 56)
    line("Step width (mean)", mean(step_widths), "m", "LOW")
    line("Shuffling steps", shuffles, "", "LOW", "{:.0f}")
    line("Freezing episodes", freeze_episodes, "", "LOW", "{:.0f}")
    line("Freezing time", freeze_time, "s", "LOW")
    print("=" * 60)
    print("  LOW = weak from a single camera; validate before clinical use.")
    print("  Step width needs the side-to-side axis the camera sees worst;")
    print("  freezing normally needs accelerometers.\n")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "skeleton_live.json"
    analyse(p)
