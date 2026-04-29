#!/usr/bin/env python3
"""Web viewer for rt/ball_state.

This debug utility subscribes to the existing BallState DDS topic and serves a
small browser UI for viewing the ball position in the robot pelvis frame.

Usage:
    python onboard/perception/debug/ball_web_viewer.py
    python onboard/perception/debug/ball_web_viewer.py --port 8090
    python onboard/perception/debug/ball_web_viewer.py --topic rt/ball_state
"""

import argparse
import json
import math
import socket
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingTCPServer

sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))

from common.ball_state_dds import (  # noqa: E402
    BALL_STATE_TOPIC,
    BALL_STALE_MS,
    SOURCE_CAM,
    SOURCE_LIDAR,
    SOURCE_NONE,
    BallState,
    BallStateSubscriber,
)
from common.lidar_ball_debug_dds import (  # noqa: E402
    LIDAR_BALL_DEBUG_TOPIC,
    LIDAR_BALL_DEBUG_STALE_MS,
    LidarBallDebugState,
    LidarBallDebugSubscriber,
)


SOURCE_NAMES = {
    SOURCE_NONE: "none",
    SOURCE_CAM: "camera",
    SOURCE_LIDAR: "lidar",
}


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ball State Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f172a;
      --panel: #111827;
      --panel2: #1e293b;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --grid: #334155;
      --axis-x: #38bdf8;
      --axis-y: #f472b6;
      --axis-z: #a3e635;
      --fresh: #22c55e;
      --camera: #38bdf8;
      --lidar: #f59e0b;
      --unknown: #a78bfa;
      --stale: #64748b;
      --bad: #ef4444;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: radial-gradient(circle at top, #1e293b, var(--bg) 55%);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 18px 24px 8px;
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }
    h1 { margin: 0; font-size: 22px; font-weight: 650; }
    .subtitle { color: var(--muted); font-size: 13px; }
    .layout {
      display: grid;
      grid-template-columns: minmax(320px, 1fr) 340px;
      gap: 16px;
      padding: 12px 24px 24px;
    }
    .card {
      background: color-mix(in srgb, var(--panel) 92%, transparent);
      border: 1px solid #334155;
      border-radius: 18px;
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.24);
      overflow: hidden;
    }
    .canvasWrap { padding: 14px; }
    canvas {
      width: 100%;
      height: min(70vh, 720px);
      display: block;
      background: #020617;
      border-radius: 12px;
    }
    .side { padding: 18px; display: grid; gap: 14px; align-content: start; }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--panel2);
      font-weight: 650;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--stale);
    }
    .kv {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 9px 16px;
      padding: 14px;
      background: rgba(15, 23, 42, 0.74);
      border: 1px solid #253247;
      border-radius: 14px;
      font-variant-numeric: tabular-nums;
    }
    .kv div:nth-child(odd) { color: var(--muted); }
    .value { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .legend { display: grid; gap: 7px; color: var(--muted); font-size: 13px; }
    .legendRow { display: flex; align-items: center; gap: 8px; }
    .swatch { width: 11px; height: 11px; border-radius: 50%; background: var(--stale); }
    code { color: #c4b5fd; }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; padding: 12px; }
      header { padding: 16px 12px 4px; }
      canvas { height: 62vh; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Ball State Viewer</h1>
      <div class="subtitle">DDS <code>__TOPIC__</code> in pelvis frame: +X forward, +Y left, +Z up</div>
    </div>
    <div class="subtitle" id="connection">connecting...</div>
  </header>
  <main class="layout">
    <section class="card canvasWrap">
      <canvas id="scene"></canvas>
    </section>
    <aside class="card side">
      <div class="status"><span class="dot" id="statusDot"></span><span id="statusText">waiting</span></div>
      <div class="kv">
        <div>x forward</div><div class="value" id="x">--</div>
        <div>y left</div><div class="value" id="y">--</div>
        <div>z up</div><div class="value" id="z">--</div>
        <div>xy distance</div><div class="value" id="distXY">--</div>
        <div>3d distance</div><div class="value" id="dist3D">--</div>
        <div>age</div><div class="value" id="age">--</div>
        <div>source</div><div class="value" id="source">--</div>
        <div>lidar raw MID360</div><div class="value" id="lidarRaw">--</div>
        <div>base y bias</div><div class="value" id="baseYBias">--</div>
        <div>lidar candidates</div><div class="value" id="lidarCounts">--</div>
        <div>timestamp</div><div class="value" id="ts">--</div>
      </div>
      <div class="legend">
        <div class="legendRow"><span class="swatch" style="background: var(--camera)"></span>camera source</div>
        <div class="legendRow"><span class="swatch" style="background: var(--lidar)"></span>lidar source</div>
        <div class="legendRow"><span class="swatch" style="background: var(--unknown)"></span>unknown/fused source</div>
        <div class="legendRow"><span class="swatch" style="background: var(--stale)"></span>invalid or stale</div>
      </div>
    </aside>
  </main>
  <script>
    const RANGE_M = __RANGE_M__;
    const HISTORY_SEC = __HISTORY_SEC__;
    const MINOR_GRID_M = 0.5;
    const MAJOR_GRID_M = 1.0;
    const canvas = document.getElementById("scene");
    const ctx = canvas.getContext("2d");
    const els = {
      connection: document.getElementById("connection"),
      statusDot: document.getElementById("statusDot"),
      statusText: document.getElementById("statusText"),
      x: document.getElementById("x"),
      y: document.getElementById("y"),
      z: document.getElementById("z"),
      distXY: document.getElementById("distXY"),
      dist3D: document.getElementById("dist3D"),
      age: document.getElementById("age"),
      source: document.getElementById("source"),
      lidarRaw: document.getElementById("lidarRaw"),
      baseYBias: document.getElementById("baseYBias"),
      lidarCounts: document.getElementById("lidarCounts"),
      ts: document.getElementById("ts"),
    };
    let state = null;
    let history = [];

    function cssVar(name) {
      return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }

    function colorFor(s) {
      if (!s || !s.fresh || !s.valid) return cssVar("--stale");
      if (s.source_name === "camera") return cssVar("--camera");
      if (s.source_name === "lidar") return cssVar("--lidar");
      if (s.source_name === "none") return cssVar("--fresh");
      return cssVar("--unknown");
    }

    function fmt(v, digits = 3, suffix = " m") {
      return Number.isFinite(v) ? `${v >= 0 ? "+" : ""}${v.toFixed(digits)}${suffix}` : "--";
    }

    function fmtPlain(v, digits = 3, suffix = " m") {
      return Number.isFinite(v) ? `${v.toFixed(digits)}${suffix}` : "--";
    }

    function fmtVec(v) {
      if (!v || !Number.isFinite(v.x) || !Number.isFinite(v.y) || !Number.isFinite(v.z)) return "--";
      return `(${fmt(v.x)}, ${fmt(v.y)}, ${fmt(v.z)})`;
    }

    function updatePanel(s) {
      const color = colorFor(s);
      els.statusDot.style.background = color;
      if (!s || !s.has_sample) {
        els.statusText.textContent = "waiting for DDS sample";
      } else if (!s.fresh) {
        els.statusText.textContent = "stale";
      } else if (!s.valid) {
        els.statusText.textContent = "no ball";
      } else {
        els.statusText.textContent = "ball visible";
      }
      els.x.textContent = s && s.has_sample ? fmt(s.x) : "--";
      els.y.textContent = s && s.has_sample ? fmt(s.y) : "--";
      els.z.textContent = s && s.has_sample ? fmt(s.z) : "--";
      els.distXY.textContent = s && s.has_sample ? fmtPlain(s.distance_xy) : "--";
      els.dist3D.textContent = s && s.has_sample ? fmtPlain(s.distance_3d) : "--";
      els.age.textContent = s && s.age_ms !== null ? `${s.age_ms.toFixed(0)} ms` : "--";
      els.source.textContent = s && s.has_sample ? `${s.source_name} (${s.source})` : "--";
      const dbg = s && s.lidar_debug && s.lidar_debug.has_sample ? s.lidar_debug : null;
      els.lidarRaw.textContent = dbg ? fmtVec(dbg.raw_mid360) : "--";
      els.baseYBias.textContent = dbg ? fmt(dbg.base_y_bias) : "--";
      els.lidarCounts.textContent = dbg ? `${dbg.candidate_count} cand / ${dbg.in_shell_count} shell` : "--";
      els.ts.textContent = s && s.timestamp_us ? new Date(s.timestamp_us / 1000).toLocaleTimeString() : "--";
    }

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(320, Math.floor(rect.width * dpr));
      const h = Math.max(320, Math.floor(rect.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function isMajorTick(m) {
      return Math.abs(m / MAJOR_GRID_M - Math.round(m / MAJOR_GRID_M)) < 1e-6;
    }

    function tickLabel(m, axis) {
      if (Math.abs(m) < 1e-6) return `${axis}=0`;
      return `${m > 0 ? "+" : ""}${m.toFixed(0)}m`;
    }

    function drawTopGrid(cx, cy, scale, width, height) {
      ctx.font = "12px ui-monospace, monospace";
      ctx.textBaseline = "middle";
      for (let m = -RANGE_M; m <= RANGE_M + 1e-6; m += MINOR_GRID_M) {
        const major = isMajorTick(m);
        ctx.strokeStyle = cssVar("--grid");
        ctx.lineWidth = major ? 1.25 : 1;
        ctx.globalAlpha = major ? 0.55 : 0.18;

        const x = cx - m * scale;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        const y = cy - m * scale;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();

        if (major) {
          ctx.globalAlpha = 0.8;
          ctx.fillStyle = cssVar("--muted");
          ctx.textAlign = "center";
          if (Math.abs(m) > 1e-6) ctx.fillText(tickLabel(m, "Y"), x, cy + 18);
          ctx.textAlign = "left";
          if (Math.abs(m) > 1e-6) ctx.fillText(tickLabel(m, "X"), cx + 10, y);
        }
      }
      ctx.globalAlpha = 1;
    }

    function drawSideGrid(originX, originY, scale, x0, y0, width, height) {
      ctx.font = "12px ui-monospace, monospace";
      ctx.textBaseline = "middle";
      ctx.fillStyle = cssVar("--muted");

      for (let m = 0; m <= RANGE_M + 1e-6; m += MINOR_GRID_M) {
        const major = isMajorTick(m);
        ctx.strokeStyle = cssVar("--grid");
        ctx.lineWidth = major ? 1.25 : 1;
        ctx.globalAlpha = major ? 0.55 : 0.18;
        const x = originX + m * scale;
        ctx.beginPath();
        ctx.moveTo(x, y0);
        ctx.lineTo(x, y0 + height);
        ctx.stroke();
        if (major) {
          ctx.globalAlpha = 0.8;
          ctx.textAlign = "center";
          ctx.fillText(tickLabel(m, "X"), x, originY + 18);
        }
      }

      for (let m = -1.0; m <= 1.5 + 1e-6; m += MINOR_GRID_M) {
        const major = isMajorTick(m);
        ctx.strokeStyle = cssVar("--grid");
        ctx.lineWidth = major ? 1.25 : 1;
        ctx.globalAlpha = major ? 0.55 : 0.18;
        const y = originY - m * scale;
        ctx.beginPath();
        ctx.moveTo(x0, y);
        ctx.lineTo(x0 + width, y);
        ctx.stroke();
        if (major && Math.abs(m) > 1e-6) {
          ctx.globalAlpha = 0.8;
          ctx.textAlign = "left";
          ctx.fillText(tickLabel(m, "Z"), originX + 10, y);
        }
      }
      ctx.globalAlpha = 1;
    }

    function drawArrow(x1, y1, x2, y2, color, label) {
      const angle = Math.atan2(y2 - y1, x2 - x1);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - 10 * Math.cos(angle - 0.5), y2 - 10 * Math.sin(angle - 0.5));
      ctx.lineTo(x2 - 10 * Math.cos(angle + 0.5), y2 - 10 * Math.sin(angle + 0.5));
      ctx.closePath();
      ctx.fill();
      ctx.font = "13px ui-monospace, monospace";
      ctx.fillText(label, x2 + 6, y2 - 6);
    }

    function drawRobot(cx, cy) {
      ctx.save();
      ctx.translate(cx, cy);
      ctx.fillStyle = "#e5e7eb";
      ctx.strokeStyle = "#94a3b8";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.roundRect(-18, -26, 36, 52, 8);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#0f172a";
      ctx.beginPath();
      ctx.arc(0, -18, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    function drawTopView(x0, y0, w, h) {
      const pad = 44;
      const cx = x0 + w / 2;
      const cy = y0 + h / 2;
      const scale = Math.min((w - 2 * pad), (h - 2 * pad)) / (RANGE_M * 2);
      ctx.save();
      ctx.beginPath();
      ctx.rect(x0, y0, w, h);
      ctx.clip();
      ctx.translate(x0, y0);
      drawTopGrid(w / 2, h / 2, scale, w, h);
      ctx.restore();
      ctx.fillStyle = cssVar("--muted");
      ctx.font = "14px system-ui, sans-serif";
      ctx.fillText("Top view: X forward, Y left", x0 + 16, y0 + 24);
      drawArrow(cx, cy, cx, cy - scale * 0.8, cssVar("--axis-x"), "+X");
      drawArrow(cx, cy, cx - scale * 0.8, cy, cssVar("--axis-y"), "+Y");
      drawRobot(cx, cy);
      drawHistory((p) => [cx - p.y * scale, cy - p.x * scale]);
      drawBall((p) => [cx - p.y * scale, cy - p.x * scale]);
    }

    function drawSideView(x0, y0, w, h) {
      const pad = 42;
      const originX = x0 + pad;
      const originY = y0 + h * 0.68;
      const scale = Math.min((w - 2 * pad) / RANGE_M, (h - 2 * pad) / RANGE_M);
      ctx.save();
      ctx.beginPath();
      ctx.rect(x0, y0, w, h);
      ctx.clip();
      drawSideGrid(originX, originY, scale, x0, y0, w, h);
      ctx.restore();
      ctx.fillStyle = cssVar("--muted");
      ctx.font = "14px system-ui, sans-serif";
      ctx.fillText("Side view: X forward, Z up", x0 + 16, y0 + 24);
      drawArrow(originX, originY, originX + scale * 0.8, originY, cssVar("--axis-x"), "+X");
      drawArrow(originX, originY, originX, originY - scale * 0.8, cssVar("--axis-z"), "+Z");
      drawHistory((p) => [originX + p.x * scale, originY - p.z * scale]);
      drawBall((p) => [originX + p.x * scale, originY - p.z * scale]);
    }

    function drawHistory(project) {
      const now = performance.now() / 1000;
      history = history.filter((p) => now - p.local_time <= HISTORY_SEC);
      for (const p of history) {
        const age = Math.max(0, Math.min(1, (now - p.local_time) / HISTORY_SEC));
        const [x, y] = project(p);
        ctx.globalAlpha = 0.15 + 0.45 * (1 - age);
        ctx.fillStyle = colorFor(p);
        ctx.beginPath();
        ctx.arc(x, y, 3 + 3 * (1 - age), 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    function drawBall(project) {
      if (!state || !state.has_sample) return;
      const [x, y] = project(state);
      ctx.fillStyle = colorFor(state);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, state.fresh && state.valid ? 12 : 8, 0, Math.PI * 2);
      ctx.fill();
      if (state.fresh && state.valid) ctx.stroke();
    }

    function draw() {
      resizeCanvas();
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      const gap = 12;
      const topH = Math.floor((rect.height - gap) * 0.62);
      drawTopView(0, 0, rect.width, topH);
      drawSideView(0, topH + gap, rect.width, rect.height - topH - gap);
      requestAnimationFrame(draw);
    }

    function handleState(s) {
      state = s;
      if (s.has_sample && s.fresh && s.valid) {
        history.push({...s, local_time: performance.now() / 1000});
      }
      updatePanel(s);
    }

    window.addEventListener("resize", resizeCanvas);
    draw();

    const events = new EventSource("/events");
    events.onopen = () => { els.connection.textContent = "connected"; };
    events.onerror = () => { els.connection.textContent = "reconnecting..."; };
    events.onmessage = (event) => handleState(JSON.parse(event.data));
  </script>
</body>
</html>
"""


class BallStateStore:
    def __init__(self, stale_ms):
        self._stale_ms = stale_ms
        self._debug_stale_ms = LIDAR_BALL_DEBUG_STALE_MS
        self._lock = threading.Lock()
        self._last = None
        self._received_at = None
        self._debug_last = None
        self._debug_received_at = None

    def update(self, sample):
        if not isinstance(sample, BallState):
            return
        with self._lock:
            self._last = sample
            self._received_at = time.monotonic()

    def update_debug(self, sample):
        if not isinstance(sample, LidarBallDebugState):
            return
        with self._lock:
            self._debug_last = sample
            self._debug_received_at = time.monotonic()

    def _debug_snapshot_unlocked(self):
        sample = self._debug_last
        received_at = self._debug_received_at
        if sample is None or received_at is None:
            return {
                "has_sample": False,
                "fresh": False,
                "valid": False,
            }
        age_ms = (time.monotonic() - received_at) * 1000.0
        fresh = age_ms <= self._debug_stale_ms
        return {
            "has_sample": True,
            "fresh": bool(fresh),
            "valid": bool(sample.valid) and bool(fresh),
            "age_ms": age_ms,
            "timestamp_us": int(sample.timestamp_us),
            "raw_mid360": {
                "x": float(sample.raw_x),
                "y": float(sample.raw_y),
                "z": float(sample.raw_z),
            },
            "kf_mid360": {
                "x": float(sample.kf_x),
                "y": float(sample.kf_y),
                "z": float(sample.kf_z),
            },
            "base": {
                "x": float(sample.base_x),
                "y": float(sample.base_y),
                "z": float(sample.base_z),
            },
            "base_y_bias": float(sample.base_y_bias),
            "candidate_count": int(sample.candidate_count),
            "in_shell_count": int(sample.in_shell_count),
        }

    def snapshot(self):
        with self._lock:
            sample = self._last
            received_at = self._received_at
            lidar_debug = self._debug_snapshot_unlocked()

        if sample is None or received_at is None:
            return {
                "has_sample": False,
                "fresh": False,
                "valid": False,
                "age_ms": None,
                "timestamp_us": 0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "distance_xy": 0.0,
                "distance_3d": 0.0,
                "source": SOURCE_NONE,
                "source_name": "none",
                "lidar_debug": lidar_debug,
            }

        age_ms = (time.monotonic() - received_at) * 1000.0
        fresh = age_ms <= self._stale_ms
        x, y, z = float(sample.x), float(sample.y), float(sample.z)
        source = int(sample.source)
        return {
            "has_sample": True,
            "fresh": bool(fresh),
            "valid": bool(sample.valid) and bool(fresh),
            "raw_valid": bool(sample.valid),
            "age_ms": age_ms,
            "timestamp_us": int(sample.timestamp_us),
            "x": x,
            "y": y,
            "z": z,
            "distance_xy": math.hypot(x, y),
            "distance_3d": math.sqrt(x * x + y * y + z * z),
            "source": source,
            "source_name": SOURCE_NAMES.get(source, "unknown"),
            "lidar_debug": lidar_debug,
        }


def _get_url(host, port):
    if host not in ("0.0.0.0", "::"):
        return f"http://{host}:{port}/"
    guess = "127.0.0.1"
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("192.168.123.1", 1))
        guess = sock.getsockname()[0]
    except OSError:
        try:
            guess = socket.gethostbyname(socket.gethostname())
        except OSError:
            pass
    finally:
        if sock is not None:
            sock.close()
    return f"http://{guess}:{port}/"


def _build_html(topic, range_m, history_sec):
    return (
        HTML_TEMPLATE
        .replace("__TOPIC__", topic)
        .replace("__RANGE_M__", json.dumps(float(range_m)))
        .replace("__HISTORY_SEC__", json.dumps(float(history_sec)))
    ).encode("utf-8")


def _make_handler(store, html, interval_s):
    class BallViewerHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send_bytes(self, body, content_type, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send_bytes(html, "text/html; charset=utf-8")
                return
            if self.path == "/state":
                body = json.dumps(store.snapshot()).encode("utf-8")
                self._send_bytes(body, "application/json")
                return
            if self.path == "/health":
                self._send_bytes(b"ok\n", "text/plain; charset=utf-8")
                return
            if self.path == "/events":
                self._serve_events()
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _serve_events(self):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    payload = json.dumps(store.snapshot(), separators=(",", ":"))
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(interval_s)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

    return BallViewerHandler


def main():
    parser = argparse.ArgumentParser(
        description="DDS rt/ball_state browser viewer"
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="HTTP bind host, default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=8090,
                        help="HTTP port, default: 8090")
    parser.add_argument("--domain-id", type=int, default=0,
                        help="CycloneDDS domain id, default: 0")
    parser.add_argument("--topic", default=BALL_STATE_TOPIC,
                        help=f"DDS topic, default: {BALL_STATE_TOPIC}")
    parser.add_argument("--lidar-debug-topic", default=LIDAR_BALL_DEBUG_TOPIC,
                        help=f"Optional LiDAR debug DDS topic, default: {LIDAR_BALL_DEBUG_TOPIC}")
    parser.add_argument("--stale-ms", type=float, default=BALL_STALE_MS,
                        help=f"Mark samples stale after this many ms, default: {BALL_STALE_MS}")
    parser.add_argument("--rate-hz", type=float, default=30.0,
                        help="Browser update rate, default: 30 Hz")
    parser.add_argument("--range-m", type=float, default=3.0,
                        help="Half-width of the top view in metres, default: 3.0")
    parser.add_argument("--history-sec", type=float, default=3.0,
                        help="Visible trail duration in seconds, default: 3.0")
    args = parser.parse_args()

    interval_s = 1.0 / max(1.0, float(args.rate_hz))
    store = BallStateStore(stale_ms=float(args.stale_ms))
    sub = BallStateSubscriber(
        domain_id=args.domain_id,
        callback=store.update,
        topic_name=args.topic,
    )
    debug_sub = LidarBallDebugSubscriber(
        domain_id=args.domain_id,
        callback=store.update_debug,
        topic_name=args.lidar_debug_topic,
    )
    sub.start()
    debug_sub.start()

    html = _build_html(args.topic, args.range_m, args.history_sec)
    handler = _make_handler(store, html, interval_s)
    ThreadingTCPServer.allow_reuse_address = True
    httpd = ThreadingTCPServer((args.host, args.port), handler)
    httpd.daemon_threads = True

    print(f"[INFO] Subscribed to DDS topic {args.topic!r} (domain={args.domain_id})")
    print(f"[INFO] Subscribed to LiDAR debug topic {args.lidar_debug_topic!r}")
    print(f"[INFO] Ball web viewer started -> open {_get_url(args.host, args.port)}")
    print("[INFO] Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        sub.stop()
        debug_sub.stop()
        httpd.shutdown()
        httpd.server_close()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
