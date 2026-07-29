# PhysioTrack — Live Gait & Joint-Angle Capture

A desktop application that lets a physiotherapist record a patient performing a
standard mobility exercise (Timed Up and Go, Sit-to-Stand, or general gait),
see live joint angles and pose tracking during the session, and get a full
spatiotemporal gait report automatically — no scripts, no terminal, no
programming knowledge required.

Built on an Intel RealSense depth camera + MediaPipe pose estimation, with a
webcam-only build available for testing and demos without the hardware.

---

## What it does

- **Tracks only the patient.** If a physiotherapist assists in frame, the app
  uses depth to tell them apart (the assisting therapist stands farther from
  the camera) and never loses the lock on the patient.
- **Never fabricates a measurement.** If a joint is occluded from the camera,
  the app reports it as `not visible` instead of guessing — a deliberate
  medical-safety rule, not a bug.
- **Live feedback during the exercise.** Real-time knee and hip angles,
  side-by-side with the raw camera feed.
- **Full gait analysis, computed automatically.** Number of steps, cadence,
  step/stride length, step time, step-to-step variability, double support,
  turn time, and steps during the turn — no manual scripting needed.
- **Organised, automatic data export.** Every session is saved to
  `sessions/<patient_id>/<exercise>/`, timestamped, with a 3D skeleton JSON,
  a per-frame angle CSV, a gait-parameter CSV, and session metadata.

---

## Repository structure

```
physiotrack/
├── physio_app_realsense.py     # Production app — Intel RealSense D455/D435
├── physio_app_webcam_full.py   # Webcam build — no depth camera required
├── gait_analysis.py            # Offline: full gait report from a saved session JSON
├── gait_diagnose.py            # Offline: per-foot speed diagnostics for threshold tuning
├── gait_calibrate.py           # Offline: sweeps stance-detection settings against a
│                                #   manually counted step total
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** the two `physio_app_*` files share almost all of their code —
> patient tracking, angle computation, gait analysis, the three-screen UI,
> and the save format are identical. The only difference is the camera
> engine (RealSense depth vs. plain webcam), so a fix or feature added to
> one should generally be mirrored in the other.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`pyrealsense2` is only needed for `physio_app_realsense.py`. If you're only
running the webcam build, you can skip installing it (see
`requirements.txt` for details).

### 2. Download the pose model

Both apps need MediaPipe's multi-person pose model. Download it and place it
**in the same folder as the scripts**:

```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

This file is intentionally **not** committed to the repository (see
`.gitignore`) because it's a large binary asset better fetched fresh.

### 3. (RealSense only) Install the camera SDK

Install the Intel RealSense SDK for your OS from
[github.com/IntelRealSense/librealsense](https://github.com/IntelRealSense/librealsense),
and confirm the camera works in the **RealSense Viewer** app before running
`physio_app_realsense.py`.

---

## Running the app

**With a RealSense camera (production, depth-accurate):**
```bash
python physio_app_realsense.py
```

**With a laptop webcam (testing/demo only, no depth):**
```bash
python physio_app_webcam_full.py
```

> ⚠️ The webcam build has no depth sensor, so joint angles come from
> MediaPipe's internal 3D estimate rather than true metric measurements. It's
> useful for testing the interface and the patient-tracking logic, but is
> **not** a substitute for the RealSense build when accuracy matters.

---

## Using the app

1. **Setup screen** — enter the patient ID, choose the exercise, enter
   height/weight and the walk-path length, and check the camera preview.
   Press **Start session**.
2. **Live screen** — the patient's skeleton is tracked and shown next to the
   raw video, with live knee/hip angles. Use **Re-lock patient** if tracking
   is ever lost. Press **End session** when the exercise is finished.
3. **Results screen** — view the computed angle summary and full gait report.
   Use **Open session folder** to see the saved files directly.

---

## Offline analysis scripts

These operate on a saved `*_skeleton_live.json` from a previous session (or
one produced by the app), and are useful for tuning and validation rather
than day-to-day clinical use:

```bash
# Full gait report from a saved session
python gait_analysis.py path/to/session_skeleton_live.json

# Per-foot speed diagnostics (use before tuning thresholds)
python gait_diagnose.py path/to/session_skeleton_live.json

# Calibrate the stance-detection threshold against a real, manually counted step total
python gait_calibrate.py path/to/session_skeleton_live.json 14
```

---

## Known limitations

- **Single camera → the far-side limb is estimated, not measured.** The
  visibility rule keeps this honest by declining to report hidden joints,
  but it doesn't recover the missing measurement. A second, front-facing
  camera is the structural fix, not yet implemented.
- **Capture frame rate affects step-count reliability.** At low effective
  frame rates (observed as low as ~12 fps under some settings), a step's
  swing phase spans very few frames, which can cause miscounts. Reducing the
  camera's streamed resolution to raise the frame rate, and re-running
  `gait_calibrate.py` afterward, is the recommended fix.
- **This is a research prototype**, not a certified medical device. Every
  metric should be validated against manual measurement (stopwatch,
  goniometer) before being relied on clinically.

---

## Acknowledgements

Built with [MediaPipe](https://github.com/google-ai-edge/mediapipe),
[PySide6](https://doc.qt.io/qtforpython/), the
[Intel RealSense SDK](https://github.com/IntelRealSense/librealsense), and
NumPy/SciPy.
