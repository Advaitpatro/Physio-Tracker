#!/usr/bin/env python3
"""
physio_app_realsense_full.py — GaitLab Pro  (RealSense / depth-sensor build)
All logic preserved.  Dark-dashboard UI identical to webcam build.

RealSense upgrades over the webcam build
  - Depth-fused metric 3D angles (accurate, not model estimates)
  - Patient locked by nearest-camera depth, not image size
  - Per-joint 3D deque smoothing via rs2_deproject_pixel_to_point
  - Spatial / temporal / hole-filling depth post-processing
  - "Save for Unity" exports depth-accurate skeleton_live.json

SETUP
    pip install PySide6 pyrealsense2 mediapipe opencv-python numpy
    # Place pose_landmarker_full.task next to this script.
RUN
    python physio_app_realsense_full.py
"""

import sys, os, re, csv, json, time, datetime
from collections import deque

import numpy as np
import cv2
import pyrealsense2 as rs
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QMargins
from PySide6.QtGui  import (QImage, QPixmap, QFont, QColor, QBrush,
                             QLinearGradient)
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit,
    QComboBox, QHBoxLayout, QVBoxLayout, QGridLayout,
    QFrame, QStackedWidget, QScrollArea, QSizePolicy,
    QGraphicsDropShadowEffect, QFileDialog,
)

try:
    from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
    HAS_CHARTS = True
except ImportError:
    HAS_CHARTS = False

# ── model & skeleton ──────────────────────────────────────────────────────────
MODEL = "pose_landmarker_full.task"

NAMES = [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER","RIGHT_EYE_INNER",
    "RIGHT_EYE","RIGHT_EYE_OUTER","LEFT_EAR","RIGHT_EAR","MOUTH_LEFT",
    "MOUTH_RIGHT","LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW",
    "LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY","RIGHT_PINKY","LEFT_INDEX",
    "RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB","LEFT_HIP","RIGHT_HIP",
    "LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL",
    "RIGHT_HEEL","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX",
]
IDX  = {n: i for i, n in enumerate(NAMES)}
CONN = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
        (23,24),(23,25),(25,27),(24,26),(26,28),(27,31),(28,32),
        (0,11),(0,12)]
ANGLES = {
    "L knee": ("LEFT_HIP",      "LEFT_KNEE",  "LEFT_ANKLE"),
    "R knee": ("RIGHT_HIP",     "RIGHT_KNEE", "RIGHT_ANKLE"),
    "L hip":  ("LEFT_SHOULDER", "LEFT_HIP",   "LEFT_KNEE"),
    "R hip":  ("RIGHT_SHOULDER","RIGHT_HIP",  "RIGHT_KNEE"),
}
VIS_THRESHOLD = 0.6

# ── design tokens — HEAT / ENERGY palette (zero blue) ────────────────────────
BG   = "#0C0D0E"    # warm near-black
C1   = "#161819"    # card surface
C2   = "#1E2224"    # elevated card
C3   = "#2C3238"    # border

CY   = "#F97316"    # orange — primary accent
CY_D = "#EA6B0A"    # orange dark (hover)
VI   = "#FCD34D"    # warm yellow
GR   = "#22C55E"    # emerald green
AM   = "#E879F9"    # magenta
RD   = "#F43F5E"    # rose-red

TX   = "#F5EFE8"    # warm near-white text
T2   = "#7A8490"    # secondary / labels
T3   = "#3E4850"    # very muted

JCOL = {"L knee": CY, "R knee": VI, "L hip": GR, "R hip": AM}

TB1  = "#080A0B"
TB2  = "#10130F"


# ═══════════════════════════════════════════════════════════════════════════════
#  GAIT COMPUTATION  (preserved exactly from webcam build)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_gait(frames, names):
    keys = ["n_steps","cadence","step_len","stride_len","step_time",
            "step_len_cv","step_time_cv","ds_t","ds_pct","turn_time","steps_in_turn"]
    res = {k: float("nan") for k in keys}
    try:
        idx = {n: i for i, n in enumerate(names)}
        n   = len(frames)
        if n < 10:
            return res
        t   = np.array([f["t"] for f in frames], float)
        fps = 1.0 / np.median(np.diff(t))

        def joint(name):
            i = idx[name]
            a = np.array([[f["px"][i], f["py"][i], f["pz"][i]] for f in frames], float)
            zero = (a[:,0]==0)&(a[:,1]==0)&(a[:,2]==0)
            a[zero] = np.nan
            ii = np.arange(n)
            for c in range(3):
                col = a[:,c]; g = ~np.isnan(col)
                if g.sum() >= 2:
                    a[:,c] = np.interp(ii, ii[g], col[g])
            return a

        def smooth(x, k=5):
            return x if len(x) < k else np.convolve(x, np.ones(k)/k, mode="same")

        def runs(mask, ml):
            out, s = [], None
            for i, m in enumerate(mask):
                if m and s is None:           s = i
                elif not m and s is not None: out.append((s,i)); s = None
            if s is not None: out.append((s, len(mask)))
            return [(a,b) for a,b in out if b-a >= ml]

        hip   = (joint("LEFT_HIP") + joint("RIGHT_HIP")) / 2.0
        hip_g = hip[:, [0,2]]
        c     = hip_g - hip_g.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        u     = vt[0]

        def ap(pg):
            return (pg - hip_g.mean(0)) @ u

        minst = max(2, int(0.10 * fps))
        foot  = {}
        for side in ("LEFT","RIGHT"):
            g  = joint(side+"_ANKLE")[:,[0,2]]
            sp = smooth(np.r_[0, np.linalg.norm(np.diff(g,axis=0),axis=1)] * fps)
            st = sp < 0.18
            iv = runs(st, minst)
            falls = [{"t": t[a], "ap": float(ap(g[(a+b)//2:(a+b)//2+1])[0]),
                      "side": side} for a,b in iv]
            foot[side] = {"stance": st, "falls": falls}

        ff = sorted(foot["LEFT"]["falls"]+foot["RIGHT"]["falls"], key=lambda f: f["t"])
        res["n_steps"] = len(ff)

        apc    = smooth(ap(hip_g), 7)
        turn_i = int(np.argmax(apc))
        fwd    = smooth(np.r_[0, np.diff(apc)] * fps, 7)
        lo     = turn_i
        while lo > 0   and fwd[lo] > -0.05 and (turn_i-lo) < fps*2.5: lo -= 1
        hi     = turn_i
        while hi < n-1 and fwd[hi] <  0.05 and (hi-turn_i) < fps*2.5: hi += 1
        res["turn_time"]     = t[hi] - t[lo]
        res["steps_in_turn"] = sum(1 for f in ff if t[lo] <= f["t"] <= t[hi])

        sl, stt = [], []
        for i in range(1, len(ff)):
            if ff[i]["side"] != ff[i-1]["side"]:
                sl.append(abs(ff[i]["ap"]-ff[i-1]["ap"]))
                stt.append(ff[i]["t"]-ff[i-1]["t"])
        strl = []
        for side in ("LEFT","RIGHT"):
            fs   = [f for f in ff if f["side"]==side]
            strl += [abs(fs[i]["ap"]-fs[i-1]["ap"]) for i in range(1,len(fs))]

        dur  = t[-1]-t[0]
        res["cadence"]      = 60.0*len(ff)/dur if dur > 0 else float("nan")
        m_   = lambda x: float(np.mean(x)) if x else float("nan")
        cv_  = lambda x: float(np.std(x)/np.mean(x)*100) if x and np.mean(x) else float("nan")
        res["step_len"]     = m_(sl);   res["stride_len"]    = m_(strl)
        res["step_time"]    = m_(stt);  res["step_len_cv"]   = cv_(sl)
        res["step_time_cv"] = cv_(stt)
        both = foot["LEFT"]["stance"] & foot["RIGHT"]["stance"]
        res["ds_t"]   = float(np.sum(both)/fps)
        res["ds_pct"] = float(100.0*np.sum(both)/n)
    except Exception as e:
        print("gait compute error:", e)
    return res


# ═══════════════════════════════════════════════════════════════════════════════
#  REALSENSE POSE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def angle3d(p3, a, b, c):
    """Angle at b using depth-fused metric 3D points {name: (x,y,z)}."""
    pa, pb, pc = p3.get(a), p3.get(b), p3.get(c)
    if pa is None or pb is None or pc is None:
        return float("nan")
    v1 = np.array(pa) - np.array(pb)
    v2 = np.array(pc) - np.array(pb)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float("nan") if den < 1e-9 else float(
        np.degrees(np.arccos(np.clip(np.dot(v1,v2)/den, -1, 1))))


def hip_center(landmarks, w, h):
    l, r = landmarks[IDX["LEFT_HIP"]], landmarks[IDX["RIGHT_HIP"]]
    return np.array([(l.x+r.x)/2*w, (l.y+r.y)/2*h])


def sample_depth(depth, px, py, w, h, win=3):
    """Median over a small window; returns 0 if no valid reading."""
    vals = []
    for dy in range(-win, win+1):
        for dx in range(-win, win+1):
            x, y = px+dx, py+dy
            if 0 <= x < w and 0 <= y < h:
                v = depth.get_distance(x, y)
                if v > 0:
                    vals.append(v)
    return float(np.median(vals)) if vals else 0.0


def person_depth(lmk, depth, w, h):
    """Median torso distance (m) — used to track patient vs physio."""
    zs = []
    for name in ("LEFT_HIP","RIGHT_HIP","LEFT_SHOULDER","RIGHT_SHOULDER"):
        i = IDX[name]
        px = min(max(int(lmk[i].x*w), 0), w-1)
        py = min(max(int(lmk[i].y*h), 0), h-1)
        d  = sample_depth(depth, px, py, w, h)
        if d > 0:
            zs.append(d)
    return float(np.median(zs)) if zs else None


# ═══════════════════════════════════════════════════════════════════════════════
#  POSE ENGINE — REALSENSE
# ═══════════════════════════════════════════════════════════════════════════════
class PoseEngine(QThread):
    frame_ready = Signal(object, object, dict, object, bool)

    def __init__(self):
        super().__init__()
        self.running        = True
        self.relock         = False
        self.patient_center = None
        self.patient_depth  = None
        self._locked        = False
        self.hist           = {k: deque(maxlen=5) for k in ANGLES}
        self.pt_hist        = {n: deque(maxlen=5) for n in NAMES}

    def lock(self):
        self.relock = True

    def run(self):
        pipe    = rs.pipeline()
        cfg     = rs.config()
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16,  30)
        cfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
        pipe.start(cfg)
        align   = rs.align(rs.stream.color)
        spatial  = rs.spatial_filter()
        temporal = rs.temporal_filter()
        hole     = rs.hole_filling_filter()

        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL),
            running_mode=vision.RunningMode.VIDEO, num_poses=2,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
        lm      = vision.PoseLandmarker.create_from_options(opts)
        last_ts = -1

        try:
            while self.running:
                fs    = pipe.wait_for_frames()
                fs    = align.process(fs)
                depth = fs.get_depth_frame()
                color = fs.get_color_frame()
                if not depth or not color:
                    continue

                depth = spatial.process(depth)
                depth = temporal.process(depth)
                depth = hole.process(depth).as_depth_frame()

                rgb  = np.asanyarray(color.get_data())
                h, w = rgb.shape[:2]
                intr = color.profile.as_video_stream_profile().intrinsics
                bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                ts = int(time.time()*1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts

                res    = lm.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                people = res.pose_landmarks

                if self.relock:
                    self.patient_center = None
                    self.patient_depth  = None
                    self._locked        = False
                    for b in self.pt_hist.values():
                        b.clear()
                    self.relock = False

                pi = None
                if people:
                    centers = [hip_center(p, w, h) for p in people]
                    depths  = [person_depth(p, depth, w, h) for p in people]

                    if self.patient_center is None:
                        # Lock to nearest person (patient walks in front of physio)
                        near = [(i, z) for i, z in enumerate(depths) if z is not None]
                        if near:
                            pi = min(near, key=lambda t: t[1])[0]
                        else:
                            tgt = np.array([w/2, h*0.6])
                            pi  = int(np.argmin([np.linalg.norm(c-tgt) for c in centers]))
                    else:
                        best, best_cost = None, 1e9
                        for i, c in enumerate(centers):
                            cost = np.linalg.norm(c - self.patient_center) / 150.0
                            if depths[i] is not None and self.patient_depth is not None:
                                cost += abs(depths[i] - self.patient_depth) / 0.5
                            if cost < best_cost:
                                best_cost, best = cost, i
                        pi = best if best_cost < 3.5 else None

                    if pi is not None:
                        self.patient_center = centers[pi]
                        if depths[pi] is not None:
                            self.patient_depth = (
                                depths[pi] if self.patient_depth is None
                                else 0.7*self.patient_depth + 0.3*depths[pi])
                        self._locked = True
                    else:
                        self._locked = False
                else:
                    self._locked = False

                overlay = bgr.copy()
                if people:
                    for k, p in enumerate(people):
                        if k == pi:
                            col, thk = (22, 115, 249), 3    # orange (BGR)
                        else:
                            col, thk = (50, 70, 85), 1
                        for a, b in CONN:
                            cv2.line(overlay,
                                     (int(p[a].x*w), int(p[a].y*h)),
                                     (int(p[b].x*w), int(p[b].y*h)), col, thk)
                        if k == pi:
                            for jt in p:
                                cv2.circle(overlay,
                                           (int(jt.x*w), int(jt.y*h)),
                                           5, (30, 140, 255), -1)

                # Depth-fused metric 3D for patient
                p3   = None
                visd = {}
                if pi is not None:
                    lmk = people[pi]
                    p3  = {}
                    for k in range(len(NAMES)):
                        name = NAMES[k]
                        vv   = getattr(lmk[k], "visibility", None)
                        visd[name] = float(vv) if vv is not None else 1.0
                        cx = min(max(int(lmk[k].x*w), 0), w-1)
                        cy = min(max(int(lmk[k].y*h), 0), h-1)
                        dd = sample_depth(depth, cx, cy, w, h)
                        buf = self.pt_hist[name]
                        if dd != 0:
                            X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], dd)
                            buf.append((X, Y, Z))
                        elif buf:
                            buf.popleft()
                        if buf:
                            m = np.median(np.array(buf), axis=0)
                            p3[name] = (float(m[0]), float(m[1]), float(m[2]))
                        else:
                            p3[name] = None

                def joint_ok(name):
                    return (p3 is not None and p3.get(name) is not None
                            and visd.get(name, 0.0) >= VIS_THRESHOLD)

                smoothed = {}
                for lab, (a, b, c) in ANGLES.items():
                    if joint_ok(a) and joint_ok(b) and joint_ok(c):
                        raw = angle3d(p3, a, b, c)
                        self.hist[lab].append(raw)
                        smoothed[lab] = float(np.median(self.hist[lab]))
                    else:
                        self.hist[lab].clear()
                        smoothed[lab] = float("nan")

                self.frame_ready.emit(bgr, overlay, smoothed, p3, self._locked)

        finally:
            pipe.stop()
            lm.close()

    def stop(self):
        self.running = False
        self.wait(1500)


def to_pixmap(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888))


# ═══════════════════════════════════════════════════════════════════════════════
#  DESIGN PRIMITIVES  (identical to webcam build)
# ═══════════════════════════════════════════════════════════════════════════════
def _glow(widget, color_hex, radius=20, alpha=80):
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(radius); eff.setOffset(0, 0)
    c = QColor(color_hex); c.setAlpha(alpha)
    eff.setColor(c); widget.setGraphicsEffect(eff)


class Divider(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFixedHeight(1)
        self.setStyleSheet(f"background:{C3}; border:none;")


class SectionHeader(QLabel):
    def __init__(self, text, accent=CY, parent=None):
        super().__init__(text.upper(), parent)
        self.setStyleSheet(
            f"color:{accent}; font-size:10px; font-weight:700; "
            f"letter-spacing:2px; border:none; background:transparent;")


class Chip(QLabel):
    _MAP = {
        "success": f"background:#0D3D2B; color:{GR}; border:1px solid #1A6644;",
        "danger":  f"background:#3D1515; color:{RD}; border:1px solid #6B2222;",
        "caution": f"background:#3D2D0A; color:{AM}; border:1px solid #6B4E14;",
        "neutral": f"background:{C2}; color:{T2}; border:1px solid {C3};",
        "primary": f"background:#0A2D33; color:{CY}; border:1px solid #155566;",
        "rec_on":  f"background:{RD}; color:#FFFFFF; border:none;",
        "depth":   f"background:#0D2D1A; color:{GR}; border:1px solid #1A5530;",
    }
    def __init__(self, text, kind="neutral", parent=None):
        super().__init__(text, parent)
        self._kind = kind; self._apply(); self.setFixedHeight(26)

    def _apply(self):
        base = self._MAP.get(self._kind, self._MAP["neutral"])
        self.setStyleSheet(
            f"{base} border-radius:13px; padding:0 12px; "
            f"font-size:11px; font-weight:600;")

    def set_kind(self, k): self._kind = k; self._apply()
    def set_text(self, t): self.setText(t)


class AngleCard(QFrame):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        self._accent = JCOL.get(label, CY)
        self.setObjectName("AC")
        self.setMinimumHeight(110)
        self.setStyleSheet(
            f"QFrame#AC{{background:{C1}; border-radius:14px; border:1px solid {C3};}}")

        root = QHBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        stripe = QFrame()
        stripe.setFixedWidth(4)
        stripe.setStyleSheet(
            f"background:{self._accent}; border-radius:4px 0 0 4px; border:none;")
        root.addWidget(stripe)

        inner = QWidget(); inner.setStyleSheet("background:transparent;")
        iv = QVBoxLayout(inner); iv.setContentsMargins(14,14,14,14); iv.setSpacing(5)

        self._lbl = QLabel(label.upper())
        self._lbl.setStyleSheet(
            f"color:{T2}; font-size:10px; font-weight:700; letter-spacing:1px; "
            f"border:none; background:transparent;")
        self._val = QLabel("—")
        self._val.setStyleSheet(
            f"color:{self._accent}; font-size:36px; font-weight:700; "
            f"font-family:'Consolas','Courier New',monospace; "
            f"border:none; background:transparent;")
        self._deg = QLabel("°")
        self._deg.setStyleSheet(
            f"color:{self._accent}; font-size:18px; border:none; background:transparent;")

        iv.addWidget(self._lbl)
        row = QHBoxLayout(); row.setSpacing(2)
        row.addWidget(self._val); row.addWidget(self._deg, 0, Qt.AlignBottom); row.addStretch()
        iv.addLayout(row)
        root.addWidget(inner)

    def set(self, val):
        if isinstance(val, float) and val != val:
            self._val.setText("no signal")
            self._val.setStyleSheet(
                f"color:{T3}; font-size:13px; font-weight:500; "
                f"font-family:'Segoe UI',Arial,sans-serif; border:none; background:transparent;")
            self._deg.hide()
        else:
            self._val.setText(f"{val:.0f}")
            self._val.setStyleSheet(
                f"color:{self._accent}; font-size:36px; font-weight:700; "
                f"font-family:'Consolas','Courier New',monospace; "
                f"border:none; background:transparent;")
            self._deg.show()


class MetricCard(QFrame):
    def __init__(self, label, unit="", na="—", large=True, accent=CY, parent=None):
        super().__init__(parent)
        self._na     = na
        self._large  = large
        self._accent = accent
        self.setObjectName("MCd")
        self.setStyleSheet(
            f"QFrame#MCd{{background:{C1}; border-radius:12px; border:1px solid {C3};}}")

        v = QVBoxLayout(self); v.setContentsMargins(16,14,16,14); v.setSpacing(5)
        self._lbl = QLabel(label.upper())
        self._lbl.setStyleSheet(
            f"color:{T2}; font-size:10px; font-weight:700; letter-spacing:1px; "
            f"border:none; background:transparent;")
        vs = "26px" if large else "20px"
        self._val = QLabel(na)
        self._val.setStyleSheet(
            f"color:{accent}; font-size:{vs}; font-weight:700; "
            f"font-family:'Consolas','Courier New',monospace; border:none; background:transparent;")
        self._unit_lbl = QLabel(unit)
        self._unit_lbl.setStyleSheet(
            f"color:{T3}; font-size:11px; border:none; background:transparent;")

        v.addWidget(self._lbl)
        row = QHBoxLayout(); row.setSpacing(5)
        row.addWidget(self._val)
        if unit: row.addWidget(self._unit_lbl, 0, Qt.AlignBottom)
        row.addStretch()
        v.addLayout(row)

    def set(self, val, fmt="{:.0f}"):
        vs = "26px" if self._large else "20px"
        if isinstance(val, float) and val != val:
            self._val.setText(self._na)
            self._val.setStyleSheet(
                f"color:{T3}; font-size:{vs}; font-weight:700; "
                f"font-family:'Consolas','Courier New',monospace; border:none; background:transparent;")
        else:
            self._val.setText(fmt.format(val))
            self._val.setStyleSheet(
                f"color:{self._accent}; font-size:{vs}; font-weight:700; "
                f"font-family:'Consolas','Courier New',monospace; border:none; background:transparent;")


# ═══════════════════════════════════════════════════════════════════════════════
#  TOP BAR
# ═══════════════════════════════════════════════════════════════════════════════
class TopBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(58)
        self.setObjectName("TB")
        self.setStyleSheet(f"QWidget#TB{{background:{TB1}; border-bottom:1px solid {C3};}}")

        h = QHBoxLayout(self); h.setContentsMargins(24,0,24,0); h.setSpacing(0)

        logo = QLabel("⬡")
        logo.setStyleSheet(f"color:{CY}; font-size:22px; border:none;")
        _glow(logo, CY, radius=16, alpha=120)

        nc = QVBoxLayout(); nc.setSpacing(0)
        prod = QLabel("GaitLab Pro")
        prod.setStyleSheet(f"color:{TX}; font-size:15px; font-weight:700; border:none;")
        tag  = QLabel("RealSense  ·  Depth-Sensor Build")
        tag.setStyleSheet(f"color:{GR}; font-size:10px; border:none;")
        nc.addWidget(prod); nc.addWidget(tag)

        h.addWidget(logo); h.addSpacing(10); h.addLayout(nc); h.addStretch()

        self._steps = []
        for i, txt in enumerate(["1  Setup", "2  Live", "3  Results"]):
            if i > 0:
                sep = QLabel("›")
                sep.setStyleSheet(f"color:{T3}; font-size:14px; border:none;")
                h.addWidget(sep); h.addSpacing(4)
            lbl = QLabel(txt)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedHeight(28); lbl.setContentsMargins(16,0,16,0)
            self._steps.append(lbl); h.addWidget(lbl)
            if i < 2: h.addSpacing(4)
        self.set_step(0)

    def set_step(self, idx):
        for i, lbl in enumerate(self._steps):
            if i == idx:
                lbl.setStyleSheet(
                    f"color:{TB1}; font-size:12px; font-weight:700; "
                    f"background:{CY}; border-radius:14px; border:none;")
            elif i < idx:
                lbl.setStyleSheet(
                    f"color:{CY}; font-size:12px; font-weight:500; "
                    f"background:transparent; border:none;")
            else:
                lbl.setStyleSheet(
                    f"color:{T3}; font-size:12px; font-weight:400; "
                    f"background:transparent; border:none;")


# ═══════════════════════════════════════════════════════════════════════════════
#  SCREEN 1 — SETUP
# ═══════════════════════════════════════════════════════════════════════════════
def _fl(text):
    l = QLabel(text)
    l.setStyleSheet(
        f"color:{T2}; font-size:11px; font-weight:600; letter-spacing:0.5px; "
        f"border:none; background:transparent;")
    return l


class SetupScreen(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main

        root = QHBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        left = QWidget(); left.setFixedWidth(376); left.setObjectName("LP")
        left.setStyleSheet(f"QWidget#LP{{background:{C1}; border-right:1px solid {C3};}}")
        lv = QVBoxLayout(left); lv.setContentsMargins(40,40,40,40); lv.setSpacing(0)

        title = QLabel("New Session")
        title.setStyleSheet(f"font-size:24px; font-weight:700; color:{TX}; border:none;")
        sub = QLabel("Enter patient details to begin recording")
        sub.setStyleSheet(f"font-size:12px; color:{T2}; margin-top:4px; border:none;")
        bar = QFrame(); bar.setFixedHeight(3)
        bar.setStyleSheet(
            f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {CY}, stop:1 transparent); border:none; border-radius:2px;")
        lv.addWidget(title); lv.addWidget(sub); lv.addSpacing(8)
        lv.addWidget(bar);   lv.addSpacing(28)

        lv.addWidget(_fl("PATIENT ID")); lv.addSpacing(6)
        self.pid = QLineEdit("P-014"); lv.addWidget(self.pid); lv.addSpacing(16)

        lv.addWidget(_fl("EXERCISE")); lv.addSpacing(6)
        self.ex = QComboBox()
        self.ex.addItems(["Timed up and go (TUG)", "Sit to stand", "Gait / walk"])
        lv.addWidget(self.ex); lv.addSpacing(16)

        hw = QHBoxLayout(); hw.setSpacing(16)
        hc = QVBoxLayout(); hc.setSpacing(6)
        hc.addWidget(_fl("HEIGHT (CM)"))
        self.height = QLineEdit("170"); hc.addWidget(self.height)
        wc = QVBoxLayout(); wc.setSpacing(6)
        wc.addWidget(_fl("WEIGHT (KG)"))
        self.weight = QLineEdit("70"); wc.addWidget(self.weight)
        hw.addLayout(hc); hw.addLayout(wc)
        lv.addLayout(hw); lv.addSpacing(16)

        lv.addWidget(_fl("WALK PATH LENGTH (M)")); lv.addSpacing(6)
        self.path = QLineEdit("3.81"); lv.addWidget(self.path); lv.addSpacing(28)

        self.cam_chip = Chip("Waiting for camera…", "neutral")
        lv.addWidget(self.cam_chip, 0, Qt.AlignLeft)
        lv.addStretch()

        self.start_btn = QPushButton("Start Session  →")
        self.start_btn.setObjectName("BtnP")
        self.start_btn.setFixedHeight(46)
        self.start_btn.clicked.connect(self._go)
        lv.addWidget(self.start_btn)
        root.addWidget(left)

        right = QWidget(); right.setObjectName("RP")
        right.setStyleSheet(f"QWidget#RP{{background:{BG};}}")
        rv = QVBoxLayout(right); rv.setContentsMargins(40,40,40,40); rv.setSpacing(10)

        cap_row = QHBoxLayout(); cap_row.setSpacing(6)
        self._cam_dot = QLabel("●")
        self._cam_dot.setStyleSheet(f"color:{T3}; font-size:9px; border:none;")
        cap_lbl = QLabel("REALSENSE PREVIEW")
        cap_lbl.setStyleSheet(
            f"color:{T2}; font-size:10px; font-weight:700; letter-spacing:1.5px; border:none;")
        self._depth_chip = Chip("Depth: initialising…", "neutral")
        cap_row.addWidget(self._cam_dot); cap_row.addWidget(cap_lbl)
        cap_row.addStretch(); cap_row.addWidget(self._depth_chip)
        rv.addLayout(cap_row)

        self.preview = QLabel("Camera initialising…")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setObjectName("PV")
        self.preview.setStyleSheet(
            f"QLabel#PV{{background:{C1}; border-radius:16px; "
            f"color:{T3}; font-size:13px; border:1px solid {C3};}}")
        self.preview.setMinimumSize(520, 420)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rv.addWidget(self.preview)
        root.addWidget(right)

    def mark_camera_ready(self):
        self._cam_dot.setStyleSheet(f"color:{GR}; font-size:9px; border:none;")
        self.cam_chip.set_text("● RealSense ready")
        self.cam_chip.set_kind("depth")
        self._depth_chip.set_text("Depth active")
        self._depth_chip.set_kind("success")

    def _go(self):
        self.main.start_session(
            self.pid.text(), self.ex.currentText(),
            self.height.text(), self.weight.text(), self.path.text())


# ═══════════════════════════════════════════════════════════════════════════════
#  SCREEN 2 — LIVE
# ═══════════════════════════════════════════════════════════════════════════════
class LiveScreen(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main     = main
        self._elapsed = 0
        self._pulse   = False
        self._timer   = QTimer(self)
        self._timer.timeout.connect(self._tick)

        root = QVBoxLayout(self); root.setContentsMargins(20,14,20,14); root.setSpacing(10)

        hdr = QHBoxLayout(); hdr.setSpacing(8)
        self.sess_lbl = QLabel("—")
        self.sess_lbl.setStyleSheet(f"font-size:14px; font-weight:700; color:{TX}; border:none;")
        self.lock_chip  = Chip("◌  Searching…", "caution")
        self.depth_chip = Chip("Depth ✓", "depth")
        self.rec_chip   = Chip("● REC", "danger")
        self.timer_lbl  = QLabel("00:00")
        self.timer_lbl.setStyleSheet(
            f"font-size:26px; font-weight:700; color:{CY}; "
            f"font-family:'Consolas','Courier New',monospace; border:none;")
        _glow(self.timer_lbl, CY, radius=12, alpha=80)

        hdr.addWidget(self.sess_lbl); hdr.addSpacing(8)
        hdr.addWidget(self.lock_chip); hdr.addWidget(self.depth_chip)
        hdr.addStretch()
        hdr.addWidget(self.rec_chip); hdr.addSpacing(14); hdr.addWidget(self.timer_lbl)
        root.addLayout(hdr)

        vids = QHBoxLayout(); vids.setSpacing(14)
        def _panel(caption):
            outer = QFrame(); outer.setObjectName("VP")
            outer.setStyleSheet(
                f"QFrame#VP{{background:#070F18; border-radius:16px; border:1px solid {C3};}}")
            fl = QVBoxLayout(outer); fl.setContentsMargins(0,0,0,0); fl.setSpacing(0)
            top = QWidget(); top.setStyleSheet("background:transparent;")
            tl  = QHBoxLayout(top); tl.setContentsMargins(14,10,14,4)
            cl  = QLabel(caption.upper())
            cl.setStyleSheet(
                f"color:{T3}; font-size:10px; font-weight:700; letter-spacing:1.4px; border:none;")
            tl.addWidget(cl); tl.addStretch()
            img = QLabel(); img.setAlignment(Qt.AlignCenter)
            img.setMinimumSize(440, 325)
            img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            img.setStyleSheet("border:none; background:transparent;")
            fl.addWidget(top); fl.addWidget(img); fl.addSpacing(10)
            return outer, img

        self.actual_wrap, self.actual = _panel("RealSense Colour")
        self.pose_wrap,   self.pose   = _panel("Pose Tracking  ·  Depth-Fused")
        vids.addWidget(self.actual_wrap); vids.addWidget(self.pose_wrap)
        root.addLayout(vids)

        ang = QHBoxLayout(); ang.setSpacing(10)
        self.cards = {}
        for label in ANGLES:
            c = AngleCard(label); self.cards[label] = c; ang.addWidget(c)
        root.addLayout(ang)

        ctrl = QHBoxLayout(); ctrl.setSpacing(10)
        self.relock_btn = QPushButton("Re-lock Patient")
        self.relock_btn.setObjectName("BtnS"); self.relock_btn.setFixedHeight(40)
        self.relock_btn.clicked.connect(main.engine.lock)
        self.end_btn = QPushButton("End Session")
        self.end_btn.setObjectName("BtnP"); self.end_btn.setFixedHeight(40)
        self.end_btn.clicked.connect(main.end_session)
        ctrl.addWidget(self.relock_btn); ctrl.addStretch(); ctrl.addWidget(self.end_btn)
        root.addLayout(ctrl)

    def start_timer(self):
        self._elapsed = 0; self.timer_lbl.setText("00:00"); self._timer.start(1000)

    def stop_timer(self):
        self._timer.stop()

    def _tick(self):
        self._elapsed += 1; m, s = divmod(self._elapsed, 60)
        self.timer_lbl.setText(f"{m:02d}:{s:02d}")
        self._pulse = not self._pulse
        if self._pulse:
            self.rec_chip.setStyleSheet(
                f"background:{RD}; color:#FFFFFF; border-radius:13px; "
                f"padding:0 12px; font-size:11px; font-weight:600; border:none;")
        else:
            self.rec_chip.setStyleSheet(
                f"background:#3D1515; color:{RD}; border-radius:13px; "
                f"padding:0 12px; font-size:11px; font-weight:600; border:1px solid #6B2222;")

    def set_locked(self, locked: bool):
        if locked:
            self.lock_chip.set_text("● Patient locked"); self.lock_chip.set_kind("success")
        else:
            self.lock_chip.set_text("◌  Searching…");   self.lock_chip.set_kind("caution")


# ═══════════════════════════════════════════════════════════════════════════════
#  SCREEN 3 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
class ResultsScreen(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main

        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"QScrollArea{{background:{BG}; border:none;}}")
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        scroll.setWidget(inner); outer.addWidget(scroll)

        v = QVBoxLayout(inner); v.setContentsMargins(40,32,40,40); v.setSpacing(0)

        self.title_lbl = QLabel("Session Results")
        self.title_lbl.setStyleSheet(
            f"font-size:24px; font-weight:700; color:{TX}; border:none; background:transparent;")
        self.sub_lbl = QLabel("")
        self.sub_lbl.setStyleSheet(
            f"font-size:13px; color:{T2}; border:none; background:transparent; margin-top:2px;")
        v.addWidget(self.title_lbl); v.addWidget(self.sub_lbl); v.addSpacing(24)

        # Depth-sensor confidence banner (green, not amber warning)
        banner = QFrame(); banner.setObjectName("BNR")
        banner.setStyleSheet(
            f"QFrame#BNR{{background:#0A1F13; border-radius:10px; "
            f"border:1px solid #1A5530; border-left:3px solid {GR};}}")
        brow = QHBoxLayout(banner); brow.setContentsMargins(16,12,16,12); brow.setSpacing(10)
        bico = QLabel("◈")
        bico.setStyleSheet(f"font-size:15px; color:{GR}; border:none; background:transparent;")
        btxt = QLabel(
            "<b>REALSENSE DEPTH MODE</b> — angles computed from metric 3D "
            "(depth-fused via rs2_deproject). Values are clinically meaningful.")
        btxt.setStyleSheet(
            f"color:#5ABF8A; font-size:12px; border:none; background:transparent;")
        btxt.setWordWrap(True)
        brow.addWidget(bico, 0, Qt.AlignTop); brow.addWidget(btxt)
        v.addWidget(banner); v.addSpacing(10)

        self.saved_lbl = QLabel()
        self.saved_lbl.setStyleSheet(
            f"font-size:12px; color:{GR}; border:none; background:transparent;")
        self.saved_lbl.setWordWrap(True)
        v.addWidget(self.saved_lbl); v.addSpacing(28)

        # Joint Angles
        v.addWidget(SectionHeader("Joint Angles  ·  Depth-Fused Metric 3D", accent=CY))
        v.addSpacing(12)
        ag = QGridLayout(); ag.setSpacing(12)
        self.acards = {}
        ac_meta = [
            ("L knee peak", "°", CY), ("R knee peak", "°", VI),
            ("L knee ROM",  "°", CY), ("R knee ROM",  "°", VI),
        ]
        for i, (k, u, ac) in enumerate(ac_meta):
            c = MetricCard(k, unit=u, na="n/a", large=True, accent=ac)
            self.acards[k] = c; ag.addWidget(c, 0, i)
        v.addLayout(ag)

        if HAS_CHARTS:
            v.addSpacing(24)
            v.addWidget(SectionHeader("Knee Angles — Full Session", accent=CY))
            v.addSpacing(12)
            chart_card = QFrame(); chart_card.setObjectName("CC")
            chart_card.setStyleSheet(
                f"QFrame#CC{{background:{C1}; border-radius:12px; border:1px solid {C3};}}")
            ccl = QVBoxLayout(chart_card); ccl.setContentsMargins(0,0,0,0)
            self._chart, self._lk_s, self._rk_s = self._make_chart()
            self._cv = QChartView(self._chart)
            self._cv.setFixedHeight(200)
            self._cv.setStyleSheet("background:transparent; border:none;")
            ccl.addWidget(self._cv); v.addWidget(chart_card)
        else:
            self._lk_s = self._rk_s = None

        v.addSpacing(28); v.addWidget(Divider()); v.addSpacing(28)

        # Gait Parameters
        v.addWidget(SectionHeader("Spatiotemporal Gait Parameters", accent=VI))
        v.addSpacing(12)
        gg = QGridLayout(); gg.setSpacing(12)
        _gm = [
            ("Number of steps", "",    "{:.0f}", True,  GR),
            ("Cadence",         "spm", "{:.0f}", True,  GR),
            ("Step length",     "m",   "{:.2f}", False, CY),
            ("Stride length",   "m",   "{:.2f}", False, CY),
            ("Step time",       "s",   "{:.2f}", False, VI),
            ("Step-length var", "%",   "{:.1f}", False, AM),
            ("Step-time var",   "%",   "{:.1f}", False, AM),
            ("Double support",  "%",   "{:.0f}", False, VI),
            ("Turn time",       "s",   "{:.1f}", False, T2),
            ("Steps in turn",   "",    "{:.0f}", False, T2),
        ]
        self.gcards = {}; self._gfmt = {}; cols = 5
        for i, (lbl, unit, fmt, lg, ac) in enumerate(_gm):
            c = MetricCard(lbl, unit=unit, na="n/a", large=lg, accent=ac)
            self.gcards[lbl] = c; self._gfmt[lbl] = fmt
            gg.addWidget(c, i // cols, i % cols)
        v.addLayout(gg); v.addSpacing(32)

        # Buttons
        btns = QHBoxLayout(); btns.setSpacing(10)
        self.open_btn = QPushButton("Open Session Folder")
        self.open_btn.setObjectName("BtnS"); self.open_btn.setFixedHeight(40)
        self.open_btn.clicked.connect(main.open_folder)
        self.unity_btn = QPushButton("Save for Unity Animation")
        self.unity_btn.setObjectName("BtnS"); self.unity_btn.setFixedHeight(40)
        self.unity_btn.clicked.connect(main.save_animation)
        self.new_btn = QPushButton("New Session  →")
        self.new_btn.setObjectName("BtnP"); self.new_btn.setFixedHeight(40)
        self.new_btn.clicked.connect(main.new_session)
        btns.addWidget(self.open_btn); btns.addWidget(self.unity_btn)
        btns.addStretch(); btns.addWidget(self.new_btn)
        v.addLayout(btns); v.addStretch()

    def _make_chart(self):
        chart = QChart()
        chart.setTitle("")
        chart.setBackgroundBrush(QBrush(QColor(C1)))
        chart.setMargins(QMargins(8,4,8,4))
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignBottom)
        lk = QLineSeries(); lk.setName("L Knee"); lk.setColor(QColor(CY))
        rk = QLineSeries(); rk.setName("R Knee"); rk.setColor(QColor(VI))
        chart.addSeries(lk); chart.addSeries(rk)
        chart.createDefaultAxes()
        return chart, lk, rk

    def populate(self, angles, gait, folder, records, pid, exercise):
        self.title_lbl.setText(f"Session — {pid}")
        ds = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
        self.sub_lbl.setText(f"{exercise}  ·  {ds}")
        self.saved_lbl.setText(f"✓  Saved to: {folder}")

        self.acards["L knee peak"].set(angles.get("lk_peak", float("nan")))
        self.acards["R knee peak"].set(angles.get("rk_peak", float("nan")))
        self.acards["L knee ROM"].set( angles.get("lk_rom",  float("nan")))
        self.acards["R knee ROM"].set( angles.get("rk_rom",  float("nan")))

        gmap = {
            "Number of steps": (gait["n_steps"],      "{:.0f}"),
            "Cadence":         (gait["cadence"],       "{:.0f}"),
            "Step length":     (gait["step_len"],      "{:.2f}"),
            "Stride length":   (gait["stride_len"],    "{:.2f}"),
            "Step time":       (gait["step_time"],     "{:.2f}"),
            "Step-length var": (gait["step_len_cv"],   "{:.1f}"),
            "Step-time var":   (gait["step_time_cv"],  "{:.1f}"),
            "Double support":  (gait["ds_pct"],        "{:.0f}"),
            "Turn time":       (gait["turn_time"],     "{:.1f}"),
            "Steps in turn":   (gait["steps_in_turn"], "{:.0f}"),
        }
        for k, (val, fmt) in gmap.items():
            self.gcards[k].set(val, fmt)

        if self._lk_s is not None and records:
            self._lk_s.clear(); self._rk_s.clear()
            for t, a in records:
                lk = a.get("L knee", float("nan"))
                rk = a.get("R knee", float("nan"))
                if lk == lk: self._lk_s.append(t, lk)
                if rk == rk: self._rk_s.append(t, rk)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s or "").strip("_") or "x"


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GaitLab Pro — RealSense Edition")
        self.resize(1160, 790); self.setMinimumSize(920, 650)

        self.engine = PoseEngine()
        self.engine.frame_ready.connect(self.on_frame)
        self.engine.start()

        self.recording    = False
        self.records      = []
        self.joint_frames = []
        self.t0           = None
        self.patient_id   = "P-014"
        self.exercise     = "TUG"
        self.height       = "170"
        self.weight       = "70"
        self.path_len     = "3.81"
        self.last_folder  = ""
        self._cam_rdy     = False

        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        self.topbar = TopBar(); outer.addWidget(self.topbar)

        self.stack   = QStackedWidget()
        self.setup   = SetupScreen(self)
        self.live    = LiveScreen(self)
        self.results = ResultsScreen(self)
        for s in (self.setup, self.live, self.results):
            self.stack.addWidget(s)
        outer.addWidget(self.stack)

    def start_session(self, pid, exercise, height, weight, path):
        self.patient_id = pid or "patient"
        self.exercise   = exercise
        self.height, self.weight, self.path_len = height, weight, path
        self.engine.lock()
        self.records, self.joint_frames, self.t0 = [], [], time.time()
        self.recording = True
        self.live.sess_lbl.setText(f"{self.patient_id}  ·  {exercise}")
        self.live.start_timer()
        self.topbar.set_step(1)
        self.stack.setCurrentWidget(self.live)

    def end_session(self):
        self.recording = False
        self.live.stop_timer()
        angles = self.angle_summary()
        gait   = compute_gait(self.joint_frames, NAMES)
        folder = self.save_all(angles, gait)
        self.results.populate(
            angles, gait, folder, self.records, self.patient_id, self.exercise)
        self.topbar.set_step(2)
        self.stack.setCurrentWidget(self.results)

    def new_session(self):
        self.topbar.set_step(0)
        self.stack.setCurrentWidget(self.setup)

    def angle_summary(self):
        s = {}
        if not self.records: return s
        def series(k):
            return np.array([a.get(k, float("nan")) for _, a in self.records])
        lk, rk = series("L knee"), series("R knee")
        if np.isfinite(lk).any():
            s["lk_peak"] = float(np.nanmin(lk)); s["lk_rom"] = float(np.nanmax(lk)-np.nanmin(lk))
        if np.isfinite(rk).any():
            s["rk_peak"] = float(np.nanmin(rk)); s["rk_rom"] = float(np.nanmax(rk)-np.nanmin(rk))
        return s

    def save_all(self, angles, gait):
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pid, ex = slug(self.patient_id), slug(self.exercise)
        folder  = os.path.join("sessions", pid, ex)
        os.makedirs(folder, exist_ok=True)
        base = f"{pid}_{ex}_{ts}"
        with open(os.path.join(folder, base+".json"), "w") as f:
            json.dump({"fps":30,"jointCount":len(NAMES),
                       "jointNames":NAMES,"frames":self.joint_frames}, f)
        with open(os.path.join(folder, base+"_angles.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["time"]+list(ANGLES.keys()))
            for t, a in self.records:
                w.writerow([round(t,3)]+[round(a.get(k,float("nan")),1) for k in ANGLES])
        with open(os.path.join(folder, base+"_gait.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["parameter","value"])
            for k, val in gait.items():
                w.writerow([k,(round(val,3) if isinstance(val,float) and val==val else "n/a")])
        with open(os.path.join(folder, base+"_info.json"), "w") as f:
            json.dump({"patient_id":self.patient_id,"exercise":self.exercise,
                       "height_cm":self.height,"weight_kg":self.weight,
                       "path_length_m":self.path_len,"timestamp":ts,
                       "frames":len(self.joint_frames),"mode":"realsense_depth"}, f, indent=2)
        self.last_folder = os.path.abspath(folder)
        return self.last_folder

    def open_folder(self):
        if self.last_folder and os.path.isdir(self.last_folder):
            try:
                os.startfile(self.last_folder)
            except AttributeError:
                import subprocess; subprocess.Popen(["xdg-open", self.last_folder])

    def save_animation(self):
        """Save depth-accurate 3D skeleton for Unity bake."""
        if not self.joint_frames:
            return
        pid = slug(self.patient_id)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save skeleton for Unity",
            f"{pid}_skeleton_live.json", "JSON (*.json)")
        if not path:
            return
        out = {"fps":30,"jointCount":len(NAMES),"jointNames":NAMES,
               "frames":self.joint_frames}
        with open(path, "w") as f:
            json.dump(out, f)

    @Slot(object, object, dict, object, bool)
    def on_frame(self, actual, overlay, angles, p3, locked):
        if not self._cam_rdy:
            self._cam_rdy = True
            self.setup.mark_camera_ready()

        cur = self.stack.currentWidget()
        if cur is self.setup:
            self.setup.preview.setPixmap(
                to_pixmap(actual).scaled(
                    self.setup.preview.size(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif cur is self.live:
            self.live.actual.setPixmap(
                to_pixmap(actual).scaled(
                    self.live.actual.size(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.live.pose.setPixmap(
                to_pixmap(overlay).scaled(
                    self.live.pose.size(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
            for k, card in self.live.cards.items():
                card.set(angles.get(k, float("nan")))
            self.live.set_locked(locked)

            if self.recording and any(v == v for v in angles.values()):
                t = time.time() - self.t0
                self.records.append((t, dict(angles)))
                if p3 is not None:
                    px, py, pz = [], [], []
                    for nm in NAMES:
                        pt = p3.get(nm)
                        if pt is None:
                            px.append(0.0); py.append(0.0); pz.append(0.0)
                        else:
                            px.append(pt[0]); py.append(-pt[1]); pz.append(pt[2])
                    self.joint_frames.append(
                        {"t": round(t,4), "px": px, "py": py, "pz": pz})

    def closeEvent(self, e):
        self.engine.stop(); e.accept()


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STYLESHEET  (identical dark theme)
# ═══════════════════════════════════════════════════════════════════════════════
STYLE = f"""
QWidget {{
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
    color: {TX};
    background: {BG};
}}
QStackedWidget, QStackedWidget > QWidget {{ background: {BG}; }}

QLineEdit {{
    border: 1px solid {C3};
    border-radius: 8px;
    padding: 9px 12px;
    background: {C2};
    color: {TX};
    font-size: 13px;
    selection-background-color: #0A2D33;
}}
QLineEdit:focus {{ border: 2px solid {CY}; padding: 8px 11px; }}

QComboBox {{
    border: 1px solid {C3};
    border-radius: 8px;
    padding: 9px 12px;
    background: {C2};
    color: {TX};
    font-size: 13px;
}}
QComboBox:focus {{ border: 2px solid {CY}; padding: 8px 11px; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {C2};
    border: 1px solid {C3};
    border-radius: 8px;
    selection-background-color: #0A2D33;
    selection-color: {CY};
    padding: 4px;
    outline: none;
}}

QPushButton#BtnP {{
    background: {CY};
    color: #05151A;
    border: none;
    border-radius: 8px;
    padding: 0 24px;
    font-size: 13px;
    font-weight: 700;
    min-height: 40px;
}}
QPushButton#BtnP:hover   {{ background: {CY_D}; }}
QPushButton#BtnP:pressed {{ background: #009DAE; }}

QPushButton#BtnS {{
    background: {C2};
    color: {T2};
    border: 1px solid {C3};
    border-radius: 8px;
    padding: 0 24px;
    font-size: 13px;
    font-weight: 600;
    min-height: 40px;
}}
QPushButton#BtnS:hover   {{ background: {C3}; color: {TX}; }}
QPushButton#BtnS:pressed {{ background: #273F58; }}

QScrollBar:vertical {{
    background: {BG};
    width: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {C3};
    border-radius: 3px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{ background: none; }}
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
