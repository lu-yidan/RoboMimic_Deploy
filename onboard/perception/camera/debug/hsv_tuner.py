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


_WEB_LOCK   = threading.Lock()
_WEB_FRAME  = b""        # current JPEG bytes
_WEB_PARAMS = dict(DEFAULTS)
_WEB_STATUS = {"detected": False, "r": 0, "fill": 0.0, "cx": 0, "cy": 0}

_HTML_PAGE = """\
<!DOCTYPE html><html lang="zh">
<head>
<meta charset="utf-8">
<title>HSV Ball Tuner</title>
<style>
  body { font-family: monospace; background: #1a1a1a; color: #eee; margin: 20px; }
  h2 { color: #7cf; margin-bottom: 10px; }
  .row { display:flex; align-items:center; margin: 6px 0; gap: 12px; }
  label { width: 160px; text-align: right; }
  input[type=range] { width: 300px; }
  .val { width: 40px; text-align: left; }
  #status { margin-top: 14px; padding: 8px 14px; border-radius: 6px;
            font-size: 1.1em; display:inline-block; }
  .det  { background: #1a4a1a; color: #4f4; }
  .miss { background: #3a1a1a; color: #f44; }
  #params { margin-top: 10px; color: #fa0; font-size: 0.9em; }
  img { border: 2px solid #444; margin-top: 14px; max-width: 100%; }
  button { margin-top: 12px; padding: 8px 20px; background:#2a6; color:#fff;
           border:none; border-radius:4px; cursor:pointer; font-size:1em; }
  button:hover { background:#3b7; }
</style>
</head>
<body>
<h2>HSV Ball Tuner — live camera</h2>
<div id="sliders"></div>
<div id="status" class="miss">no ball</div>
<div id="params"></div>
<br><button onclick="saveParams()">Save params (hsv_params.txt)</button>
<br><img id="frame" src="/frame.jpg" alt="frame">
<script>
const DEFS = {H_LOW:__H_LOW__,H_HIGH:__H_HIGH__,S_MIN:__S_MIN__,V_MIN:__V_MIN__,
              DILATION:__DILATION__,FILL_MIN:__FILL_MIN__,MIN_R:__MIN_R__};
const META = {
  H_LOW:   {label:"H_LOW  (hue ≥)",  max:179},
  H_HIGH:  {label:"H_HIGH (hue ≤)",  max:179},
  S_MIN:   {label:"S_MIN  (sat ≥)",  max:255},
  V_MIN:   {label:"V_MIN  (val ≥)",  max:255},
  DILATION:{label:"DILATION (px)",   max:51},
  FILL_MIN:{label:"FILL_MIN (%)",    max:30},
  MIN_R:   {label:"MIN_R   (px)",    max:40},
};
const div = document.getElementById("sliders");
for (const [k, m] of Object.entries(META)) {
  const row = document.createElement("div"); row.className = "row";
  row.innerHTML = `<label>${m.label}</label>
    <input type="range" id="sl_${k}" min="0" max="${m.max}" value="${DEFS[k]}"
           oninput="update('${k}', this.value)">
    <span class="val" id="v_${k}">${DEFS[k]}</span>`;
  div.appendChild(row);
}
function update(k, val) {
  document.getElementById("v_"+k).textContent = val;
  const params = {};
  for (const key of Object.keys(META))
    params[key] = document.getElementById("sl_"+key).value;
  fetch("/params?" + new URLSearchParams(params));
}
function refreshFrame() {
  document.getElementById("frame").src = "/frame.jpg?t=" + Date.now();
}
function refreshStatus() {
  fetch("/status").then(r=>r.json()).then(d=>{
    const el = document.getElementById("status");
    const pe = document.getElementById("params");
    if (d.detected) {
      el.textContent = `DETECTED  r=${d.r}px  fill=${d.fill}%  cx=${d.cx}  cy=${d.cy}`;
      el.className = "det";
    } else {
      el.textContent = "no ball"; el.className = "miss";
    }
    pe.textContent = d.cmd;
  });
}
function saveParams() {
  fetch("/save").then(r=>r.json()).then(d=>alert("Saved: " + d.cmd));
}
setInterval(refreshFrame,  200);
setInterval(refreshStatus, 300);
</script>
</body></html>
"""


def run_web(port: int):
    """Web-based HSV tuner — no display needed. Open http://<robot>:port/ in a browser."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[tuner] pyrealsense2 not available")
        sys.exit(1)

    # ── camera thread ───────────────────────────────────────────────────────
    def _camera_loop():
        pipe = rs.pipeline()
        cfg  = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipe.start(cfg)
        print(f"[tuner] Camera started. Open http://0.0.0.0:{port}/")
        for _ in range(5):          # warm-up
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
                mask, merged, best = _detect(
                    frame, p["H_LOW"], p["H_HIGH"], p["S_MIN"], p["V_MIN"],
                    p["DILATION"], p["FILL_MIN"], p["MIN_R"])
                p1 = _make_panel(frame,  f"Original")
                p2 = _make_panel(mask,   f"HSV mask  H=[{p['H_LOW']},{p['H_HIGH']}] S≥{p['S_MIN']} V≥{p['V_MIN']}")
                p3 = _make_panel(merged, f"Merged (dilation={p['DILATION']}px)")
                p4 = _draw_result(frame, best)
                canvas = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])
                _, jpg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with _WEB_LOCK:
                    global _WEB_FRAME, _WEB_STATUS
                    _WEB_FRAME = jpg.tobytes()
                    if best:
                        _WEB_STATUS = {"detected": True,
                                       "r": round(best[3]), "fill": round(best[4]*100, 1),
                                       "cx": best[1], "cy": best[2]}
                    else:
                        _WEB_STATUS = {"detected": False, "r": 0, "fill": 0.0, "cx": 0, "cy": 0}
        finally:
            pipe.stop()

    t = threading.Thread(target=_camera_loop, daemon=True)
    t.start()
    time.sleep(1.5)     # let camera warm up before serving

    # ── HTTP handler ────────────────────────────────────────────────────────
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass    # silence access log

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

            elif path == "/frame.jpg":
                with _WEB_LOCK:
                    data = bytes(_WEB_FRAME)
                if not data:
                    self._send(503, "text/plain", "not ready")
                else:
                    self._send(200, "image/jpeg", data)

            elif path == "/params":
                qs = parse_qs(parsed.query)
                with _WEB_LOCK:
                    for k in ("H_LOW","H_HIGH","S_MIN","V_MIN","DILATION","FILL_MIN","MIN_R"):
                        if k in qs:
                            _WEB_PARAMS[k] = int(qs[k][0])
                self._send(200, "application/json", '{"ok":true}')

            elif path == "/status":
                import json
                with _WEB_LOCK:
                    st = dict(_WEB_STATUS)
                    p  = dict(_WEB_PARAMS)
                st["cmd"] = _params_str(p["H_LOW"], p["H_HIGH"], p["S_MIN"], p["V_MIN"])
                self._send(200, "application/json", json.dumps(st))

            elif path == "/save":
                import json
                with _WEB_LOCK:
                    p = dict(_WEB_PARAMS)
                cmd = _params_str(p["H_LOW"], p["H_HIGH"], p["S_MIN"], p["V_MIN"])
                with open("hsv_params.txt", "w") as f:
                    f.write(cmd + "\n")
                print(f"[tuner] Saved: {cmd}")
                self._send(200, "application/json", json.dumps({"cmd": cmd}))

            else:
                self._send(404, "text/plain", "not found")

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"[tuner] Web tuner at http://0.0.0.0:{port}/  (Ctrl-C to quit)")
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
