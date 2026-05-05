#!/usr/bin/env python3
from __future__ import annotations
"""Interactive HSV ball-detection tuner.

Loads images from a file or directory and shows four panels in real time:
  [Original]  [HSV mask]  [Merged (dilated)]  [Detection result]

Sliders let you tune all parameters without restarting. Press:
  n / p   — next / previous image (if directory)
  s       — save current params to hsv_params.txt
  q / Esc — quit

Usage:
    python debug/hsv_tuner.py debug/test/image.png
    python debug/hsv_tuner.py debug/test/          # cycles through all images
    python debug/hsv_tuner.py --camera              # live from RealSense D455 (needs display)
    python debug/hsv_tuner.py --web                 # web UI at http://<robot>:8092/
    python debug/hsv_tuner.py --headless            # save frames to /tmp/hsv_tuner_out/

Output: prints best --ball-hsv-* flags to paste into your launch command.
"""

import argparse
import glob
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

# ── Default HSV params (tuned for blue-purple soccer ball patches) ────────────
DEFAULTS = {
    # ── HSV mode ──────────────────────────────────────────────────────────
    "H_LOW":       100,
    "H_HIGH":      135,
    "S_MIN":        70,
    "V_MIN":        80,
    "DILATION":     17,
    "FILL_MIN":     10,
    "MIN_R":         4,
    "EXCL_TOP":     20,
    "CIRC_MIN":     40,
    # ── Hough mode ────────────────────────────────────────────────────────
    "HOUGH":         0,   # 0=HSV mode, 1=Hough circle mode
    "HOUGH_P2":     25,   # accumulator threshold (lower=more detections, noisier)
    "HOUGH_BLUR":    9,   # Gaussian blur kernel (must be odd)
    "HOUGH_MINR":        15,   # min ball radius (px)
    "HOUGH_MAXR":       100,   # max ball radius (px)
    "HOUGH_MIN_CONTRAST": 15,  # min (inner - outer annulus) brightness; ball > carpet
}

WIN = "HSV Tuner"
PANEL_W = 640   # width of each sub-panel
PANEL_H = 360


def _detect(frame, h_low, h_high, s_min, v_min, dilation, fill_min_pct, min_r,
            exclude_top_frac=0.0, circ_min=0.0):
    """Run the same algorithm as _detect_ball_hsv() in apriltag_detector.py."""
    hsv_low  = np.array([h_low,  s_min, v_min], dtype=np.uint8)
    hsv_high = np.array([h_high, 255,   255  ], dtype=np.uint8)

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    if exclude_top_frac > 0:
        cutoff = int(mask.shape[0] * exclude_top_frac)
        mask[:cutoff, :] = 0

    dil = max(3, dilation | 1)          # ensure odd
    k_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))
    merged  = cv2.dilate(mask, k_merge)
    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    merged  = cv2.morphologyEx(merged, cv2.MORPH_OPEN, k_clean)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill_min = fill_min_pct / 100.0
    dil_half = dil // 2
    h, w = mask.shape
    best = None

    for cnt in contours:
        # ── circularity: 4π·area/perimeter² (circle=1.0, square≈0.785) ──
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue
        cnt_area = cv2.contourArea(cnt)
        circularity = 4 * np.pi * cnt_area / (perimeter * perimeter)
        if circularity < circ_min:
            continue

        (bx, by), br = cv2.minEnclosingCircle(cnt)
        bx, by, br = int(bx), int(by), float(br)
        r_est = br - dil_half
        if r_est < min_r:
            continue
        circle_area = np.pi * br * br
        y0, y1 = max(0, by - int(br)), min(h, by + int(br) + 1)
        x0, x1 = max(0, bx - int(br)), min(w, bx + int(br) + 1)
        roi = mask[y0:y1, x0:x1]
        orig_px = int(np.count_nonzero(roi))
        fill = orig_px / circle_area
        if fill < fill_min:
            continue
        score = orig_px * fill * circularity   # circularity boosts rounder blobs
        if best is None or score > best[0]:
            best = (score, bx, by, r_est, fill, circularity)

    return mask, merged, best


def _detect_hough(frame, blur_k, param2, min_r, max_r, min_contrast=15,
                  exclude_top_frac=0.0):
    """Detect ball by circular shape + local contrast (inner vs outer annulus).

    Score = mean_inside - mean_outside_annulus.
    A gray/white ball on dark carpet has high contrast; a bright floor
    patch surrounded by equally-bright carpet has low contrast.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if exclude_top_frac > 0:
        gray[:int(gray.shape[0] * exclude_top_frac), :] = 0

    k = max(3, (blur_k // 2) * 2 + 1)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(min_r * 2, 30),
        param1=60,
        param2=param2,
        minRadius=min_r,
        maxRadius=max_r,
    )

    h, w = gray.shape
    vis = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    best = None

    if circles is not None:
        circles_int = np.round(circles[0]).astype(int)
        scored = []
        for cx, cy, r in circles_int:
            inner = np.zeros((h, w), np.uint8)
            cv2.circle(inner, (cx, cy), max(1, r), 255, -1)
            outer = np.zeros((h, w), np.uint8)
            cv2.circle(outer, (cx, cy), max(1, int(r * 1.6)), 255, -1)
            annulus = cv2.subtract(outer, inner)   # ring outside the ball
            mean_in  = float(cv2.mean(gray, mask=inner)[0])
            mean_out = float(cv2.mean(gray, mask=annulus)[0]) if annulus.any() else mean_in
            contrast = mean_in - mean_out          # positive = brighter inside = ball-like
            scored.append((contrast, mean_in, int(cx), int(cy), int(r)))

        # Draw all candidates: green border = passes contrast, red = fails
        for contrast, mean_in, cx, cy, r in scored:
            ok = contrast >= min_contrast
            color = (0, 200, 60) if ok else (40, 40, 200)
            cv2.circle(vis, (cx, cy), r, color, 2)
            cv2.putText(vis, f"+{contrast:.0f}" if contrast >= 0 else f"{contrast:.0f}",
                        (cx - 16, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        # Winner = highest contrast among passing candidates
        valid = [(c, mi, cx, cy, r) for c, mi, cx, cy, r in scored if c >= min_contrast]
        if valid:
            best_contrast, mean_in, bx, by, br = max(valid, key=lambda x: x[0])
            cv2.circle(vis, (bx, by), br, (0, 255, 255), 3)   # cyan = winner
            best = (0, bx, by, br, round(mean_in / 255, 2), round(best_contrast / 255, 2))

    return blurred, vis, best


def _make_panel(img, label, w=PANEL_W, h=PANEL_H):
    """Resize + label a panel."""
    panel = cv2.resize(img, (w, h))
    if panel.ndim == 2:                      # grayscale → BGR
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (200, 200, 200), 2, cv2.LINE_AA)
    return panel


def _draw_result(frame, best, w=PANEL_W, h=PANEL_H):
    """Draw detection circle on a resized copy."""
    orig_h, orig_w = frame.shape[:2]
    sx, sy = w / orig_w, h / orig_h
    panel = cv2.resize(frame.copy(), (w, h))
    if best is not None:
        _, cx, cy, r_est, fill, circ = best
        px, py = int(cx * sx), int(cy * sy)
        pr     = max(2, int(r_est * (sx + sy) / 2))
        cv2.circle(panel, (px, py), pr, (0, 255, 80), 2)
        cv2.circle(panel, (px, py), 3,  (0, 255, 80), -1)
        label = f"r={r_est:.0f}px fill={fill*100:.0f}% circ={circ:.2f}"
        cv2.putText(panel, label, (px - pr, py - pr - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 2, cv2.LINE_AA)
    else:
        cv2.putText(panel, "no detection", (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 60, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "Result", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (200, 200, 200), 2, cv2.LINE_AA)
    return panel


def _params_str(h_low, h_high, s_min, v_min):
    return (f"--ball-hsv "
            f"--ball-hsv-h-low {h_low} --ball-hsv-h-high {h_high} "
            f"--ball-hsv-s-min {s_min} --ball-hsv-v-min {v_min}")


def run_images(paths: list[str]):
    idx = 0
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, PANEL_W * 2, PANEL_H * 2)

    # Sliders
    cv2.createTrackbar("H_LOW   (hue>=)",  WIN, DEFAULTS["H_LOW"],       179, lambda v: None)
    cv2.createTrackbar("H_HIGH  (hue<=)",  WIN, DEFAULTS["H_HIGH"],      179, lambda v: None)
    cv2.createTrackbar("S_MIN   (sat>=)",  WIN, DEFAULTS["S_MIN"],       255, lambda v: None)
    cv2.createTrackbar("V_MIN   (val>=)",  WIN, DEFAULTS["V_MIN"],       255, lambda v: None)
    cv2.createTrackbar("DILATION (px)",    WIN, DEFAULTS["DILATION"],     51, lambda v: None)
    cv2.createTrackbar("FILL_MIN (%)",     WIN, DEFAULTS["FILL_MIN"],     30, lambda v: None)
    cv2.createTrackbar("MIN_R    (px)",    WIN, DEFAULTS["MIN_R"],        40, lambda v: None)
    cv2.createTrackbar("EXCL_TOP (%)",     WIN, DEFAULTS["EXCL_TOP"],     50, lambda v: None)
    cv2.createTrackbar("CIRC_MIN (x100)", WIN, DEFAULTS["CIRC_MIN"],     100, lambda v: None)

    print(f"[tuner] {len(paths)} image(s) loaded. n/p=next/prev  s=save  q=quit")

    while True:
        path = paths[idx % len(paths)]
        frame = cv2.imread(path)
        if frame is None:
            print(f"[tuner] Cannot read {path}")
            idx += 1
            continue

        h_low  = cv2.getTrackbarPos("H_LOW   (hue>=)",  WIN)
        h_high = cv2.getTrackbarPos("H_HIGH  (hue<=)",  WIN)
        s_min  = cv2.getTrackbarPos("S_MIN   (sat>=)",  WIN)
        v_min  = cv2.getTrackbarPos("V_MIN   (val>=)",  WIN)
        dil    = cv2.getTrackbarPos("DILATION (px)",    WIN)
        fill   = cv2.getTrackbarPos("FILL_MIN (%)",     WIN)
        min_r  = max(1, cv2.getTrackbarPos("MIN_R    (px)",  WIN))
        excl   = cv2.getTrackbarPos("EXCL_TOP (%)",     WIN)
        circ   = cv2.getTrackbarPos("CIRC_MIN (x100)", WIN)

        mask, merged, best = _detect(frame, h_low, h_high, s_min, v_min, dil, fill, min_r,
                                     exclude_top_frac=excl/100.0, circ_min=circ/100.0)

        p1 = _make_panel(frame,  f"Original  [{os.path.basename(path)}]")
        p2 = _make_panel(mask,   f"HSV mask  H=[{h_low},{h_high}] S>={s_min} V>={v_min}")
        p3 = _make_panel(merged, f"Merged (dilation={dil}px)")
        p4 = _draw_result(frame, best)

        top = np.hstack([p1, p2])
        bot = np.hstack([p3, p4])
        canvas = np.vstack([top, bot])

        # Status bar at bottom
        status = _params_str(h_low, h_high, s_min, v_min)
        detected = "DETECTED" if best else "no ball"
        cv2.putText(canvas, f"{detected}   {status}",
                    (8, canvas.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200) if best else (80, 80, 255),
                    1, cv2.LINE_AA)

        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('n'):
            idx += 1
        elif key == ord('p'):
            idx = max(0, idx - 1)
        elif key == ord('s'):
            out = _params_str(h_low, h_high, s_min, v_min)
            with open("hsv_params.txt", "w") as f:
                f.write(out + "\n")
            print(f"[tuner] Saved: {out}")

    cv2.destroyAllWindows()
    h_low  = cv2.getTrackbarPos("H_LOW   (hue≥)", WIN) if cv2.getWindowProperty(WIN, 0) >= 0 else DEFAULTS["H_LOW"]
    print("\n[tuner] Final params to paste:")
    print(f"  {_params_str(h_low, cv2.getTrackbarPos('H_HIGH  (hue≤)', WIN) if True else 0, 0, 0)}")


def run_camera():
    """Grab live frames from RealSense D455 (color stream only)."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[tuner] pyrealsense2 not available; use --image instead")
        sys.exit(1)

    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    print("[tuner] Camera started at 1280x720@30fps (color only)")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, PANEL_W * 2, PANEL_H * 2)
    cv2.createTrackbar("H_LOW   (hue≥)", WIN, DEFAULTS["H_LOW"],    179, lambda v: None)
    cv2.createTrackbar("H_HIGH  (hue≤)", WIN, DEFAULTS["H_HIGH"],   179, lambda v: None)
    cv2.createTrackbar("S_MIN   (sat≥)", WIN, DEFAULTS["S_MIN"],    255, lambda v: None)
    cv2.createTrackbar("V_MIN   (val≥)", WIN, DEFAULTS["V_MIN"],    255, lambda v: None)
    cv2.createTrackbar("DILATION (px)",   WIN, DEFAULTS["DILATION"],  51, lambda v: None)
    cv2.createTrackbar("FILL_MIN (%)",    WIN, DEFAULTS["FILL_MIN"],  30, lambda v: None)
    cv2.createTrackbar("MIN_R    (px)",   WIN, DEFAULTS["MIN_R"],     40, lambda v: None)
    cv2.createTrackbar("EXCL_TOP (%)",    WIN, DEFAULTS["EXCL_TOP"],  50, lambda v: None)
    cv2.createTrackbar("CIRC_MIN (x100)",WIN, DEFAULTS["CIRC_MIN"], 100, lambda v: None)

    print("[tuner] Live mode. s=save params  q=quit")
    try:
        while True:
            frames = pipe.wait_for_frames()
            cf = frames.get_color_frame()
            if not cf:
                continue
            frame = np.asanyarray(cf.get_data())

            h_low  = cv2.getTrackbarPos("H_LOW   (hue>=)",  WIN)
            h_high = cv2.getTrackbarPos("H_HIGH  (hue<=)",  WIN)
            s_min  = cv2.getTrackbarPos("S_MIN   (sat>=)",  WIN)
            v_min  = cv2.getTrackbarPos("V_MIN   (val>=)",  WIN)
            dil    = cv2.getTrackbarPos("DILATION (px)",    WIN)
            fill   = cv2.getTrackbarPos("FILL_MIN (%)",     WIN)
            min_r  = max(1, cv2.getTrackbarPos("MIN_R    (px)", WIN))
            excl   = cv2.getTrackbarPos("EXCL_TOP (%)",     WIN)
            circ   = cv2.getTrackbarPos("CIRC_MIN (x100)",  WIN)

            mask, merged, best = _detect(frame, h_low, h_high, s_min, v_min, dil, fill, min_r,
                                         exclude_top_frac=excl/100.0, circ_min=circ/100.0)

            p1 = _make_panel(frame,  "Original (live)")
            p2 = _make_panel(mask,   f"HSV mask  H=[{h_low},{h_high}] S>={s_min} V>={v_min}")
            p3 = _make_panel(merged, f"Merged (dil={dil})")
            p4 = _draw_result(frame, best)

            canvas = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])
            status = _params_str(h_low, h_high, s_min, v_min)
            cv2.putText(canvas, ("DETECTED  " if best else "no ball  ") + status,
                        (8, canvas.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 255, 200) if best else (80, 80, 255), 1, cv2.LINE_AA)
            cv2.imshow(WIN, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                out = _params_str(h_low, h_high, s_min, v_min)
                with open("hsv_params.txt", "w") as f:
                    f.write(out + "\n")
                print(f"[tuner] Saved: {out}")
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


_WEB_LOCK   = threading.Lock()
_WEB_FRAME  = b""        # current JPEG bytes (web-sized composite)
_WEB_PARAMS = dict(DEFAULTS)
_WEB_STATUS = {"detected": False, "r": 0, "fill": 0.0, "cx": 0, "cy": 0}

# Web panels are smaller than desktop panels to reduce JPEG size and latency
_WEB_PW, _WEB_PH = 480, 270   # per-panel size for web (→ 960×540 total canvas)

_HTML_PAGE = """\
<!DOCTYPE html><html lang="zh">
<head>
<meta charset="utf-8">
<title>HSV Ball Tuner</title>
<style>
  body { font-family: monospace; background: #1a1a1a; color: #eee; margin: 16px; }
  h2 { color: #7cf; margin-bottom: 8px; }
  .row { display:flex; align-items:center; margin: 5px 0; gap: 10px; }
  label { width: 200px; text-align:right; color:#adf; }
  .hint { color:#666; font-size:0.82em; width:260px; }
  input[type=range] { width: 220px; }
  .val { width: 36px; text-align:left; color:#ff8; }
  #status { margin-top:12px; padding:8px 14px; border-radius:6px;
            font-size:1.05em; display:inline-block; }
  .det  { background:#1a4a1a; color:#4f4; }
  .miss { background:#3a1a1a; color:#f44; }
  #params { margin-top:8px; color:#fa0; font-size:0.88em; word-break:break-all; }
  img { border:2px solid #444; margin-top:12px; max-width:100%; display:block; }
  button { margin-top:10px; padding:8px 22px; background:#2a6; color:#fff;
           border:none; border-radius:4px; cursor:pointer; font-size:1em; }
  button:hover { background:#3b7; }
  #fps { color:#888; font-size:0.85em; margin-left:12px; }
</style>
</head>
<body>
<h2>HSV Ball Tuner — live camera (MJPEG)</h2>
<div id="sliders"></div>
<div>
  <div id="status" class="miss">waiting...</div>
  <span id="fps"></span>
</div>
<div id="params"></div>
<br><button onclick="saveParams()">💾 Save params → hsv_params.txt</button>
<br><img src="/stream" alt="live stream">
<script>
const DEFS = {H_LOW:__H_LOW__,H_HIGH:__H_HIGH__,S_MIN:__S_MIN__,V_MIN:__V_MIN__,
              DILATION:__DILATION__,FILL_MIN:__FILL_MIN__,MIN_R:__MIN_R__,
              EXCL_TOP:__EXCL_TOP__,CIRC_MIN:__CIRC_MIN__,
              HOUGH:__HOUGH__,HOUGH_P2:__HOUGH_P2__,HOUGH_BLUR:__HOUGH_BLUR__,
              HOUGH_MINR:__HOUGH_MINR__,HOUGH_MAXR:__HOUGH_MAXR__,
              HOUGH_MIN_CONTRAST:__HOUGH_MIN_CONTRAST__};
const META_HSV = [
  {k:"H_LOW",    label:"H_LOW  颜色下限",    max:179, hint:"颜色色相最小值（蓝=100，紫=120，绿=40）"},
  {k:"H_HIGH",   label:"H_HIGH 颜色上限",    max:179, hint:"颜色色相最大值，范围越窄越精准"},
  {k:"S_MIN",    label:"S_MIN  饱和度",      max:255, hint:"★最重要★ 过低会把灰色/白色误判；建议>=60"},
  {k:"V_MIN",    label:"V_MIN  亮度",        max:255, hint:"过低会抓阴影；光线好时可调高"},
  {k:"DILATION", label:"DILATION 膨胀",      max:51,  hint:"把散碎色块合并成圆；太大会合并噪点"},
  {k:"FILL_MIN", label:"FILL_MIN 填充率%",   max:30,  hint:"越高越严格，要求圆内颜色覆盖越多"},
  {k:"MIN_R",    label:"MIN_R  最小半径",    max:40,  hint:"小于此像素的检测结果忽略"},
  {k:"CIRC_MIN", label:"CIRC_MIN 圆形度x100",max:100, hint:"圆=100，矩形=78；提高可排除屏幕/线缆"},
];
const META_HOUGH = [
  {k:"HOUGH_P2",           label:"HOUGH_P2 灵敏度",      max:80,  hint:"越低越灵敏(更多红绿候选)，越高越严格；从30开始"},
  {k:"HOUGH_BLUR",         label:"HOUGH_BLUR 模糊",       max:31,  hint:"平滑噪点；越大越平滑；奇数"},
  {k:"HOUGH_MINR",         label:"HOUGH_MINR 最小半径",   max:80,  hint:"球最小半径(px)；排除小噪点"},
  {k:"HOUGH_MAXR",         label:"HOUGH_MAXR 最大半径",   max:300, hint:"球最大半径(px)；排除过大误检"},
  {k:"HOUGH_MIN_CONTRAST", label:"MIN_CONTRAST 局部对比", max:80,  hint:"★关键★ 圆内比圆外亮多少；绿=通过，红=未通过，青=最终结果。球比地毯亮，调到刚好排除地板"},
];
const META_COMMON = [
  {k:"EXCL_TOP", label:"EXCL_TOP 排除顶部%", max:50, hint:"屏蔽画面顶部N%区域（排除屏幕/天花板干扰）"},
];
let houghMode = DEFS["HOUGH"] === 1;
function buildSliders() {
  const div = document.getElementById("sliders");
  div.innerHTML = "";
  const addRow = (m) => {
    const row = document.createElement("div"); row.className = "row";
    row.innerHTML = `<label>${m.label}</label>
      <input type="range" id="sl_${m.k}" min="0" max="${m.max}" value="${DEFS[m.k]}"
             oninput="update('${m.k}',this.value)">
      <span class="val" id="v_${m.k}">${DEFS[m.k]}</span>
      <span class="hint">${m.hint}</span>`;
    div.appendChild(row);
  };
  const modeRow = document.createElement("div"); modeRow.className = "row";
  modeRow.innerHTML = `<label style="color:#fa0">检测模式</label>
    <button onclick="toggleMode()" id="modeBtn" style="padding:4px 14px;font-size:0.9em">
      ${houghMode?"🔵 Hough圆检测":"🎨 HSV色块"}</button>
    <span class="hint">${houghMode?"灰度圆形检测，适合灰白球":"HSV颜色过滤，适合彩色色块"}</span>`;
  div.appendChild(modeRow);
  const meta = houghMode ? META_HOUGH : META_HSV;
  for (const m of meta) addRow(m);
  for (const m of META_COMMON) addRow(m);
}
buildSliders();
function toggleMode() {
  houghMode = !houghMode;
  DEFS["HOUGH"] = houghMode ? 1 : 0;
  buildSliders();
  sendAll();
}
function sendAll() {
  const p = {HOUGH: houghMode ? 1 : 0};
  const allMeta = [...META_HSV, ...META_HOUGH, ...META_COMMON];
  for (const m of allMeta) {
    const el = document.getElementById("sl_"+m.k);
    if (el) p[m.k] = parseInt(el.value);
  }
  fetch("/params?" + new URLSearchParams(p));
}
function update(k, val) {
  document.getElementById("v_"+k).textContent = val;
  DEFS[k] = parseInt(val);
  sendAll();
}
let _fps_frames = 0, _fps_t0 = Date.now();
function refreshStatus() {
  fetch("/status").then(r=>r.json()).then(d=>{
    const el = document.getElementById("status");
    if (d.detected) {
      const modeTag = d.mode ? `[${d.mode}] ` : "";
      el.textContent = `✅ ${modeTag}DETECTED  r=${d.r}px  circ=${d.circ}  cx=${d.cx}  cy=${d.cy}`;
      el.className = "det";
    } else {
      el.textContent = "❌ no ball"; el.className = "miss";
    }
    document.getElementById("params").textContent = d.cmd;
    _fps_frames++;
    const now = Date.now();
    if (now - _fps_t0 > 2000) {
      document.getElementById("fps").textContent =
        `  stream ~${Math.round(_fps_frames*1000/(now-_fps_t0))} fps`;
      _fps_frames = 0; _fps_t0 = now;
    }
  });
}
function saveParams() {
  fetch("/save").then(r=>r.json()).then(d=>alert("Saved:\\n" + d.cmd));
}
setInterval(refreshStatus, 250);
</script>
</body></html>
"""


def run_web(port: int):
    """Web-based HSV tuner — no display needed. Open http://<robot>:port/ in a browser."""
    import json as _json
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[tuner] pyrealsense2 not available")
        sys.exit(1)

    # ── camera + render thread ──────────────────────────────────────────────
    def _camera_loop():
        pipe = rs.pipeline()
        cfg  = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipe.start(cfg)
        print(f"[tuner] Camera started. Open http://0.0.0.0:{port}/  (Ctrl-C to quit)")
        for _ in range(5):
            pipe.wait_for_frames()
        try:
            while True:
                frames = pipe.wait_for_frames()
                cf = frames.get_color_frame()
                if not cf:
                    continue
                frame = np.asanyarray(cf.get_data()).copy()
                with _WEB_LOCK:
                    p = dict(_WEB_PARAMS)

                # ── always run BOTH algorithms ──────────────────────────────
                excl = p["EXCL_TOP"] / 100.0
                mask, merged, best_hsv = _detect(
                    frame, p["H_LOW"], p["H_HIGH"], p["S_MIN"], p["V_MIN"],
                    p["DILATION"], p["FILL_MIN"], p["MIN_R"],
                    exclude_top_frac=excl, circ_min=p["CIRC_MIN"]/100.0)
                gray_blur, hough_vis, best_hough = _detect_hough(
                    frame, p["HOUGH_BLUR"], p["HOUGH_P2"],
                    p["HOUGH_MINR"], p["HOUGH_MAXR"],
                    min_contrast=p["HOUGH_MIN_CONTRAST"],
                    exclude_top_frac=excl)

                best = best_hough if p["HOUGH"] else best_hsv
                mode_label = "HOUGH" if p["HOUGH"] else "HSV"

                # ── 6-panel layout: 2 rows × 3 cols ────────────────────────
                # Row 0: Original | HSV mask | Merged
                # Row 1: Grayscale| Hough vis| Combined result
                pw, ph = _WEB_PW, _WEB_PH

                # combined result: draw both on one frame
                result_frame = frame.copy()
                orig_h, orig_w = frame.shape[:2]
                sx, sy = orig_w / pw, orig_h / ph
                if best_hsv is not None:
                    hcx = int(best_hsv[1] / sx * pw / pw * orig_w)   # keep full-res coords
                    cv2.circle(result_frame, (best_hsv[1], best_hsv[2]),
                               max(2, int(best_hsv[3])), (0, 255, 60), 2)
                    cv2.putText(result_frame, f"HSV r={best_hsv[3]:.0f}",
                                (best_hsv[1] - 40, best_hsv[2] - int(best_hsv[3]) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 60), 2)
                if best_hough is not None:
                    cv2.circle(result_frame, (best_hough[1], best_hough[2]),
                               max(2, int(best_hough[3])), (0, 255, 255), 2)
                    cv2.putText(result_frame, f"Hough r={best_hough[3]:.0f}",
                                (best_hough[1] - 40, best_hough[2] + int(best_hough[3]) + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                r0 = np.hstack([
                    _make_panel(frame,       "Original", pw, ph),
                    _make_panel(mask,        f"HSV mask S>={p['S_MIN']} V>={p['V_MIN']}", pw, ph),
                    _make_panel(merged,      f"Merged dil={p['DILATION']}px", pw, ph),
                ])
                r1 = np.hstack([
                    _make_panel(gray_blur,   f"Grayscale blur={p['HOUGH_BLUR']}px", pw, ph),
                    _make_panel(hough_vis,   f"Hough p2={p['HOUGH_P2']} contrast>={p['HOUGH_MIN_CONTRAST']}", pw, ph),
                    _make_panel(result_frame,f"Result  [primary={mode_label}]", pw, ph),
                ])
                canvas = np.vstack([r0, r1])
                _, jpg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 70])
                with _WEB_LOCK:
                    global _WEB_FRAME, _WEB_STATUS
                    _WEB_FRAME = jpg.tobytes()
                    st = {"mode": mode_label}
                    if best:
                        st.update({"detected": True, "r": round(best[3]),
                                   "fill": round(best[4]*100, 1), "circ": round(best[5], 2),
                                   "cx": best[1], "cy": best[2]})
                    else:
                        st.update({"detected": False, "r": 0, "fill": 0.0,
                                   "circ": 0.0, "cx": 0, "cy": 0})
                    _WEB_STATUS = st
        finally:
            pipe.stop()

    threading.Thread(target=_camera_loop, daemon=True).start()
    time.sleep(1.5)

    # ── HTTP handler (threaded so MJPEG doesn't block /params) ─────────────
    from http.server import ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, ctype, body):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            path   = parsed.path

            if path == "/":
                html = _HTML_PAGE
                for k, v in DEFAULTS.items():
                    html = html.replace(f"__{k}__", str(v))
                self._send(200, "text/html; charset=utf-8", html)

            elif path == "/stream":
                # MJPEG push — stays open until client disconnects
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=mjpegframe")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while True:
                        with _WEB_LOCK:
                            data = bytes(_WEB_FRAME)
                        if data:
                            hdr = (b"--mjpegframe\r\n"
                                   b"Content-Type: image/jpeg\r\n"
                                   b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n")
                            self.wfile.write(hdr + data + b"\r\n")
                            self.wfile.flush()
                        time.sleep(1 / 20)   # cap at 20 fps to save bandwidth
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            elif path == "/frame.jpg":
                with _WEB_LOCK:
                    data = bytes(_WEB_FRAME)
                self._send(200 if data else 503,
                           "image/jpeg" if data else "text/plain",
                           data or b"not ready")

            elif path == "/params":
                qs = parse_qs(parsed.query)
                with _WEB_LOCK:
                    for k in ("H_LOW","H_HIGH","S_MIN","V_MIN","DILATION","FILL_MIN","MIN_R",
                              "EXCL_TOP","CIRC_MIN","HOUGH","HOUGH_P2","HOUGH_BLUR",
                              "HOUGH_MINR","HOUGH_MAXR","HOUGH_MIN_CONTRAST"):
                        if k in qs:
                            _WEB_PARAMS[k] = int(qs[k][0])
                self._send(200, "application/json", '{"ok":true}')

            elif path == "/status":
                with _WEB_LOCK:
                    st = dict(_WEB_STATUS)
                    p  = dict(_WEB_PARAMS)
                st["cmd"] = _params_str(p["H_LOW"], p["H_HIGH"], p["S_MIN"], p["V_MIN"])
                self._send(200, "application/json", _json.dumps(st))

            elif path == "/save":
                with _WEB_LOCK:
                    p = dict(_WEB_PARAMS)
                cmd = _params_str(p["H_LOW"], p["H_HIGH"], p["S_MIN"], p["V_MIN"])
                with open("hsv_params.txt", "w") as f:
                    f.write(cmd + "\n")
                print(f"[tuner] Saved: {cmd}")
                self._send(200, "application/json", _json.dumps({"cmd": cmd}))

            else:
                self._send(404, "text/plain", "not found")

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[tuner] Web tuner at http://0.0.0.0:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def run_headless(n_frames: int, out_dir: str):
    """Capture N frames from camera, save detection results as images (no display needed)."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[tuner] pyrealsense2 not available")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipe.start(cfg)
    print(f"[tuner] Headless: capturing {n_frames} frame(s) → {out_dir}/")

    # warm-up
    for _ in range(5):
        pipe.wait_for_frames()

    p = DEFAULTS
    h_low, h_high = p["H_LOW"], p["H_HIGH"]
    s_min, v_min  = p["S_MIN"],  p["V_MIN"]
    dil, fill, min_r = p["DILATION"], p["FILL_MIN"], p["MIN_R"]

    try:
        for i in range(n_frames):
            frames = pipe.wait_for_frames()
            cf = frames.get_color_frame()
            if not cf:
                continue
            frame = np.asanyarray(cf.get_data()).copy()

            mask, merged, best = _detect(frame, h_low, h_high, s_min, v_min, dil, fill, min_r)

            p1 = _make_panel(frame,  f"Original  [frame {i}]")
            p2 = _make_panel(mask,   f"HSV mask  H=[{h_low},{h_high}] S>={s_min} V>={v_min}")
            p3 = _make_panel(merged, f"Merged (dilation={dil}px)")
            p4 = _draw_result(frame, best)
            canvas = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])

            status = _params_str(h_low, h_high, s_min, v_min)
            detected = "DETECTED" if best else "no ball"
            cv2.putText(canvas, f"{detected}   {status}",
                        (8, canvas.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 255, 200) if best else (80, 80, 255), 1, cv2.LINE_AA)

            fname = os.path.join(out_dir, f"frame_{i:03d}.png")
            cv2.imwrite(fname, canvas)
            print(f"[tuner] {fname}  {detected}" + (f"  r={best[3]:.0f}px fill={best[4]*100:.0f}%" if best else ""))
    finally:
        pipe.stop()

    print(f"\n[tuner] Done. scp unitree@<robot>:{os.path.abspath(out_dir)}/*.png .")


def main():
    ap = argparse.ArgumentParser(description="Interactive HSV ball-detection tuner")
    ap.add_argument("path", nargs="?", default=None,
                    help="Image file or directory of images")
    ap.add_argument("--camera", action="store_true",
                    help="Use live RealSense D455 feed instead of images")
    ap.add_argument("--headless", action="store_true",
                    help="Camera mode without display: save N result images and exit")
    ap.add_argument("--frames", type=int, default=5,
                    help="Number of frames to capture in --headless mode (default: 5)")
    ap.add_argument("--out-dir", default="/tmp/hsv_tuner_out",
                    help="Output directory for --headless images (default: /tmp/hsv_tuner_out)")
    ap.add_argument("--web", action="store_true",
                    help="Web UI tuner — no display needed, open http://<robot>:8092/")
    ap.add_argument("--port", type=int, default=8092,
                    help="Port for --web mode (default: 8092)")
    args = ap.parse_args()

    if args.web:
        run_web(args.port)
        return

    if args.headless:
        run_headless(args.frames, args.out_dir)
        return

    if args.camera:
        run_camera()
        return

    if args.path is None:
        ap.print_help()
        sys.exit(1)

    p = args.path
    if os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, "*.png")) +
                       glob.glob(os.path.join(p, "*.jpg")) +
                       glob.glob(os.path.join(p, "*.jpeg")))
        if not paths:
            print(f"[tuner] No images found in {p}")
            sys.exit(1)
    else:
        paths = [p]

    run_images(paths)


if __name__ == "__main__":
    main()
