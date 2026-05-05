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
    python debug/hsv_tuner.py --camera              # live from RealSense D455

Output: prints best --ball-hsv-* flags to paste into your launch command.
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

# ── Default HSV params (tuned for blue-purple soccer ball patches) ────────────
DEFAULTS = {
    "H_LOW":    95,
    "H_HIGH":   135,
    "S_MIN":    30,
    "V_MIN":    130,
    "DILATION": 17,   # must be odd
    "FILL_MIN": 5,    # percent (0-30)
    "MIN_R":    4,    # pixels
}

WIN = "HSV Tuner"
PANEL_W = 640   # width of each sub-panel
PANEL_H = 360


def _detect(frame, h_low, h_high, s_min, v_min, dilation, fill_min_pct, min_r):
    """Run the same algorithm as _detect_ball_hsv() in apriltag_detector.py."""
    hsv_low  = np.array([h_low,  s_min, v_min], dtype=np.uint8)
    hsv_high = np.array([h_high, 255,   255  ], dtype=np.uint8)

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)

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
        score = orig_px * fill
        if best is None or score > best[0]:
            best = (score, bx, by, r_est, fill)

    return mask, merged, best


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
        _, cx, cy, r_est, fill = best
        px, py = int(cx * sx), int(cy * sy)
        pr     = max(2, int(r_est * (sx + sy) / 2))
        cv2.circle(panel, (px, py), pr, (0, 255, 80), 2)
        cv2.circle(panel, (px, py), 3,  (0, 255, 80), -1)
        label = f"r={r_est:.0f}px  fill={fill*100:.0f}%"
        cv2.putText(panel, label, (px - pr, py - pr - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2, cv2.LINE_AA)
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
    cv2.createTrackbar("H_LOW   (hue≥)",   WIN, DEFAULTS["H_LOW"],    179, lambda v: None)
    cv2.createTrackbar("H_HIGH  (hue≤)",   WIN, DEFAULTS["H_HIGH"],   179, lambda v: None)
    cv2.createTrackbar("S_MIN   (sat≥)",   WIN, DEFAULTS["S_MIN"],    255, lambda v: None)
    cv2.createTrackbar("V_MIN   (val≥)",   WIN, DEFAULTS["V_MIN"],    255, lambda v: None)
    cv2.createTrackbar("DILATION (px)",    WIN, DEFAULTS["DILATION"],  51, lambda v: None)
    cv2.createTrackbar("FILL_MIN (%)",     WIN, DEFAULTS["FILL_MIN"],  30, lambda v: None)
    cv2.createTrackbar("MIN_R    (px)",    WIN, DEFAULTS["MIN_R"],     40, lambda v: None)

    print(f"[tuner] {len(paths)} image(s) loaded. n/p=next/prev  s=save  q=quit")

    while True:
        path = paths[idx % len(paths)]
        frame = cv2.imread(path)
        if frame is None:
            print(f"[tuner] Cannot read {path}")
            idx += 1
            continue

        h_low  = cv2.getTrackbarPos("H_LOW   (hue≥)", WIN)
        h_high = cv2.getTrackbarPos("H_HIGH  (hue≤)", WIN)
        s_min  = cv2.getTrackbarPos("S_MIN   (sat≥)", WIN)
        v_min  = cv2.getTrackbarPos("V_MIN   (val≥)", WIN)
        dil    = cv2.getTrackbarPos("DILATION (px)",  WIN)
        fill   = cv2.getTrackbarPos("FILL_MIN (%)",   WIN)
        min_r  = max(1, cv2.getTrackbarPos("MIN_R    (px)", WIN))

        mask, merged, best = _detect(frame, h_low, h_high, s_min, v_min, dil, fill, min_r)

        p1 = _make_panel(frame,  f"Original  [{os.path.basename(path)}]")
        p2 = _make_panel(mask,   f"HSV mask  H=[{h_low},{h_high}] S≥{s_min} V≥{v_min}")
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
    cv2.createTrackbar("DILATION (px)",  WIN, DEFAULTS["DILATION"],  51, lambda v: None)
    cv2.createTrackbar("FILL_MIN (%)",   WIN, DEFAULTS["FILL_MIN"],  30, lambda v: None)
    cv2.createTrackbar("MIN_R    (px)",  WIN, DEFAULTS["MIN_R"],     40, lambda v: None)

    print("[tuner] Live mode. s=save params  q=quit")
    try:
        while True:
            frames = pipe.wait_for_frames()
            cf = frames.get_color_frame()
            if not cf:
                continue
            frame = np.asanyarray(cf.get_data())

            h_low  = cv2.getTrackbarPos("H_LOW   (hue≥)", WIN)
            h_high = cv2.getTrackbarPos("H_HIGH  (hue≤)", WIN)
            s_min  = cv2.getTrackbarPos("S_MIN   (sat≥)", WIN)
            v_min  = cv2.getTrackbarPos("V_MIN   (val≥)", WIN)
            dil    = cv2.getTrackbarPos("DILATION (px)",  WIN)
            fill   = cv2.getTrackbarPos("FILL_MIN (%)",   WIN)
            min_r  = max(1, cv2.getTrackbarPos("MIN_R    (px)", WIN))

            mask, merged, best = _detect(frame, h_low, h_high, s_min, v_min, dil, fill, min_r)

            p1 = _make_panel(frame,  "Original (live)")
            p2 = _make_panel(mask,   f"HSV mask  H=[{h_low},{h_high}] S≥{s_min} V≥{v_min}")
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
            p2 = _make_panel(mask,   f"HSV mask  H=[{h_low},{h_high}] S≥{s_min} V≥{v_min}")
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
    args = ap.parse_args()

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
